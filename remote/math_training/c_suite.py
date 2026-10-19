"""Sixteen-hour C-focused suite: interpolation, format controls, NLL and confirmation."""
import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

from core import read,write,sha,digest,jsonl
from c_utils import (POLICY,make_plan,budget_plan,reshuffle_plan,category,select,
                     rank_key,outcome_intervals,grade_diagnostics)
from published_data import response_tokens
from published_suite import PublishedSuite
from suite import Suite,PhaseFailure,PhaseTimeout,link_dir
from suite_utils import immutable_json,summarize,candidate_report

PRIOR_CODE={
    'published_suite.py':'e4b0308588ca9f44594b5d18cc4d20e5b7c8f12a8e7efbede72641aa27ef72d7',
    'worker.py':'59d2381fe12445ff892d92e130aa0c2581fda12399b5be4963139884b7450c61',
    'published_data.py':'5a819948ef747ede080480c91d6f0a4cfb474a8920737ec70cfb2381f4007c93'}


class CSuite(PublishedSuite):
    def __init__(self,args):
        args.source_suite=args.source_suite.expanduser().resolve()
        root=args.workspace.expanduser().resolve()/'suites'/args.run_id
        args.config_overrides=dict(context=8192,generations=1,eval_chunk=8,dtype=args.dtype,
            published_terminal=True,auto_oom_microbatch=True,suite_kind='c_balance_v1',
            c_source_suite=str(args.source_suite),c_public_data=str(root/'published_data'))
        Suite.__init__(self,args)
        self.prior=args.source_suite
        self.data=self.root/'published_data'
        self.form_data=self.root/'format_data'
        self.summary.update(selection_policy=POLICY,experiment='C-focused NLL; interpolation and format controls',
            uncertainty_scope='Repeated dev exploration; question bootstrap excludes execution variability')

    def stage(self,name,minutes,reserve):
        self.cutoff=min(self.deadline-reserve*60,time.time()+minutes*60)
        self.summary['stage']=name
        self.save()

    def verify_prior(self):
        from worker import check_model
        ctx=read(self.prior/'suite_context.json')
        for name,expected in ctx['code'].items():
            if sha(self.prior/'code'/name)!=expected:
                raise ValueError('Prior code snapshot changed: '+name)
        for name,expected in PRIOR_CODE.items():
            if ctx['code'].get(name)!=expected:
                raise ValueError('Unsupported prior public suite implementation: '+name)
        for name in ('input_manifest.json','published_sources.json','chat_template.jinja','grading.py'):
            if ctx['code'][name]!=self.code[name]:
                raise ValueError('Prior input/grading contract differs: '+name)
        for key in ('base_model','base_revision','stage2_sha256','dtype','seed','context',
                    'max_tokens','generations','eval_chunk','vllm_memory'):
            if ctx['config'][key]!=self.cfg[key]:
                raise ValueError('Prior model/evaluation setting differs: '+key)
        root=self.prior/'training_trials/C_gsm_limo_nll'
        old_ctx=read(root/'context.json'); plan=read(root/'schedule.json')
        signature=digest([old_ctx['signature'],sha(root/'pairs.json'),None,digest(plan)])
        if read(root/'training/rft/complete.json')['signature']!=signature:
            raise ValueError('Prior C training data/schedule identity changed')
        for kind in ('quarter','half'):
            adapter=root/f'training/rft/milestones/step-{plan["milestone_steps"][kind]:05d}'
            if not check_model(adapter): raise ValueError('Missing prior C milestone: '+kind)
        immutable_json(self.root/'prior_manifest.json',{'suite':str(self.prior),
            'context_sha256':sha(self.prior/'suite_context.json'),'pairs_sha256':sha(root/'pairs.json'),
            'schedule_sha256':sha(root/'schedule.json'),
            'adapters':{kind:sha(root/f'training/rft/milestones/step-{plan["milestone_steps"][kind]:05d}/adapter_model.safetensors')
                        for kind in ('quarter','half')}})

    def initialize(self):
        self.verify_prior()
        PublishedSuite.initialize(self)
        for name in ('gsm8k_pairs.json','limo_pairs.json','limo_dev.json','limo_test.json'):
            if read(self.data/name)!=read(self.prior/'published_data'/name):
                raise ValueError('Prior public data/split differs: '+name)
        old=self.prior/'training_trials/C_gsm_limo_nll'
        rows,plan=read(old/'pairs.json'),read(old/'schedule.json')
        if rows!=self.gsm+self.limo: raise ValueError('Prior C dataset order differs')
        self.old_plan=plan
        root=self.root/'training_trials/old_C_recipe'
        old_cfg=read(old/'context.json')['config']
        self.training_setup(root,rows,make_plan(rows,plan['groups'],milestones=plan['milestone_steps']),
                            learning_rate=old_cfg['learning_rate'],microbatch=old_cfg['microbatch'])
        self.old_recipe_root=root
        diag={'prior_training':make_plan(rows,plan['groups'],milestones=plan['milestone_steps']),
              'grader':'unchanged; no automatic alternate-answer matching',
              'prior_outputs':{},'successful_training_seconds':{}}
        diag['prior_training'].pop('groups'); diag['prior_training'].pop('cumulative_tokens')
        for kind in ('quarter','half','full'):
            p=self.prior/f'evaluations/C_gsm_limo_nll_{kind}/generations/rft_eval.json'
            diag['prior_outputs'][kind]=grade_diagnostics(read(p),self.dev)
        diag['prior_outputs']['b0']=grade_diagnostics(read(self.prior/'shared/generations/b0_eval.json'),self.dev)
        for name in ('A_gsm_nll','B_gsm_dft','C_gsm_limo_nll','D_gsm_limo_dft'):
            p=self.prior/f'training_trials/{name}/training/rft/train.jsonl'
            records=jsonl(p) if p.exists() else []
            diag['successful_training_seconds'][name]=sum(r['seconds'] for r in records)
        immutable_json(self.root/'diagnosis.json',diag)
        ctx=self.context(self.form_data,self.cfg)
        self.command([self.args.train_python,str(self.snapshot/'c_worker.py'),'prepare',
                      '--context',str(ctx)],'prepare_format_controls')
        self.summary['diagnosis']=diag
        self.summary['format_data']=read(self.form_data/'format_report.json')
        self.save()

    def training_setup(self,root,rows,plan,learning_rate=5e-6,microbatch=1):
        self.share_models(root)
        immutable_json(root/'pairs.json',rows); immutable_json(root/'schedule.json',plan)
        cfg=dict(self.cfg,objective='rft',published_schedule=True,published_mixed=True,
                 train_seed=plan['seed'],learning_rate=learning_rate,microbatch=microbatch)
        return self.context(root,cfg)

    def train_until(self,root,step,tag):
        return self.command([self.args.train_python,'-u',str(self.snapshot/'worker.py'),'train',
            '--context',str(root/'context.json'),'--candidate','rft','--until-step',str(step)],tag)

    def main_timed(self,kind,limit_seconds,operation):
        """Separate training/evaluation wall-time caps, including failed operations."""
        usage=self.summary.setdefault('main_wall_seconds',{})
        remaining=limit_seconds-usage.get(kind,0)
        if remaining<=45: raise PhaseTimeout('Main '+kind+' time allowance exhausted')
        cutoff=self.cutoff; started=time.time()
        self.cutoff=min(cutoff,started+remaining)
        try: return operation()
        finally:
            usage[kind]=usage.get(kind,0)+time.time()-started
            self.cutoff=cutoff
            self.save()

    def adapter(self,root,step):
        return root/f'training/rft/milestones/step-{step:05d}'

    def interpolate(self,name,right,alpha):
        destination=self.root/'interpolations'/name
        self.command([self.args.train_python,str(self.snapshot/'c_worker.py'),'interpolate',
            '--left',str(self.inputs/'stage2'),'--right',str(right),'--destination',str(destination),
            '--alpha',str(alpha)],'interpolate_'+name)
        return destination

    def evaluate(self,name,adapter,recipe,selectable=True,count=400,purpose='eval',slot='evaluations'):
        root=self.root/slot/name
        table='trials' if slot=='evaluations' and count==400 and purpose=='eval' else slot
        registry=self.summary.setdefault(table,{})
        old=registry.get(name)
        if old and old.get('status')=='complete': return old
        self.share_models(root)
        link_dir(adapter,root/'training/rft/final')
        ctx=self.context(root,self.cfg)
        immutable_json(root/'eval_rows.json',self.dev)
        immutable_json(root/'final_rows.json',self.final)
        rows=(self.dev if purpose=='eval' else self.final)[:count]
        baseline=self.base_outputs if purpose=='eval' else read(self.shared/'generations/b0_final.json')
        baseline={r['id']:baseline[r['id']] for r in rows}
        result={'status':'running','path':str(root),'adapter':str(adapter),
                'recipe':recipe,'selectable':selectable,'train_seed':recipe.get('seed',42)}
        registry[name]=result; self.save()
        try:
            self.phase(ctx,'merge',branch='rft',tag=slot+'_'+name+'_merge')
            self.phase(ctx,'generate',branch='rft',purpose=purpose,count=count,tag=slot+'_'+name+'_'+purpose)
            outputs=read(root/f'generations/rft_{purpose}.json')
            result.update(candidate_report(baseline,outputs,rows),status='complete')
            result['outcome_intervals']=outcome_intervals(baseline,outputs)
            result['grade_diagnostics']=grade_diagnostics(outputs,rows)
            if count==400:
                result['category']=category(summarize(baseline,rows),result['metrics'])
                result['limo_'+('dev' if purpose=='eval' else 'test')]=self.auxiliary(root,'rft','dev' if purpose=='eval' else 'test')
        except (PhaseFailure,PhaseTimeout) as exc:
            result.update(status='time_limited' if isinstance(exc,PhaseTimeout) else 'failed',reason=str(exc))
        finally:
            self.remove_merged(root,'rft'); self.save()
        print('RESULT',name,result['status'],result.get('metrics',{}).get('accuracy'),flush=True)
        return result

    def interpolation_stage(self):
        # Verify alpha=0 after merging against the shared b0, in addition to tensor endpoint checks.
        right=self.prior/f'training_trials/C_gsm_limo_nll/training/rft/milestones/step-{self.old_plan["milestone_steps"]["half"]:05d}'
        zero=self.interpolate('endpoint_zero',right,0.)
        self.interpolate('endpoint_one',right,1.)
        audit=self.root/'interpolation_audit'; self.share_models(audit)
        link_dir(zero,audit/'training/rft/final'); ctx=self.context(audit,self.cfg)
        if not (audit/'integrity.json').exists():
            self.phase(ctx,'merge',branch='rft'); self.phase(ctx,'audit_weights')
            self.remove_merged(audit,'rft')
        if not read(audit/'integrity.json')['identical']:
            raise ValueError('Interpolation alpha=0 merged model differs from b0')
        for kind in ('quarter','half'):
            step=self.old_plan['milestone_steps'][kind]
            adapter=self.prior/f'training_trials/C_gsm_limo_nll/training/rft/milestones/step-{step:05d}'
            recipe={'training':str(self.old_recipe_root),'step':step,'alpha':1.,'kind':'old_C','seed':42}
            self.evaluate('old_C_'+kind,adapter,recipe)
            for alpha in (.25,.5,.75):
                name=f'C_{kind}_alpha{int(100*alpha)}'
                mixed=self.interpolate(name,adapter,alpha)
                self.evaluate(name,mixed,dict(recipe,alpha=alpha,kind='interpolation'))

    def formats_stage(self):
        for form in ('F00','F10','F01','F11'):
            root=self.root/'training_trials'/form
            plan=read(self.form_data/form/'schedule.json')
            self.training_setup(root,read(self.form_data/form/'pairs.json'),plan)
            record=self.summary.setdefault('training',{}).setdefault(form,{'status':'running'})
            try:
                half=plan['milestone_steps']['half']
                self.train_until(root,half,form+'_train_half')
                recipe={'training':str(root),'step':half,'alpha':1.,'kind':'format','form':form,'seed':42}
                self.evaluate(form+'_half',self.adapter(root,half),recipe,count=128,slot='quick_formats')
                self.train_until(root,len(plan['groups']),form+'_train_full')
                recipe.update(step=len(plan['groups']))
                self.evaluate(form+'_full',self.adapter(root,len(plan['groups'])),recipe)
                record['status']='complete'
            except (PhaseFailure,PhaseTimeout) as exc:
                record.update(status='time_limited' if isinstance(exc,PhaseTimeout) else 'failed',reason=str(exc))
            self.save()
        path=self.root/'format_choice.json'
        if path.exists(): return read(path)['form']
        trials={k:v for k,v in self.summary['trials'].items() if k in {f'{f}_full' for f in ('F00','F10','F01','F11')}}
        best=select(self.summary['baseline'],trials)
        chosen='F00'; reason='no clear passing alternative to raw response + EOS'
        if best:
            control=trials.get('F00_full',{})
            if control.get('status')=='complete' and trials[best]['metrics']['correct']>control['metrics']['correct']+4:
                chosen=best[:3]; reason='passing format with more than four additional dev correct answers'
        immutable_json(path,{'form':chosen,'reason':reason,'selected_before_main_training':True})
        return chosen

    def profile(self,rows,plan):
        root=self.root/'benchmarks/main'; self.share_models(root)
        immutable_json(root/'pairs.json',rows); immutable_json(root/'schedule.json',plan)
        path=root/'profile.json'
        if path.exists(): return read(path)
        cfg=dict(self.cfg,objective='rft',published_schedule=True,published_mixed=True,train_seed=42)
        measurements=[]
        for microbatch in (1,2,4):
            try:
                mem=self.context(root,dict(cfg,microbatch=microbatch,benchmark_mode='longest',benchmark_steps=1),f'memory_{microbatch}.json')
                self.phase(mem,'bench',branch='rft',tag=f'longest_memory_mb{microbatch}')
                speed=self.context(root,dict(cfg,microbatch=microbatch,benchmark_mode='representative',benchmark_steps=6),f'speed_{microbatch}.json')
                self.phase(speed,'bench',branch='rft',tag=f'representative_speed_mb{microbatch}')
                measurements.append(read(root/'timings/bench_rft.json'))
            except PhaseFailure as exc:
                write(root/f'failure_{microbatch}.json',{'error':str(exc)})
                if 'out of memory' not in str(exc).lower(): raise
                break
        if not measurements: raise PhaseFailure('No microbatch fits')
        result=min(measurements,key=lambda x:x['seconds_per_update'])
        immutable_json(path,dict(result,measurements=measurements))
        return read(path)

    def main_stage(self,form):
        rows=read(self.form_data/form/'pairs.json')
        path=self.root/'main_budget.json'
        started=time.time()
        if path.exists(): budget=read(path)
        else:
            capacity=min(3*sum(response_tokens(r) for r in rows if r['dataset']=='gsm8k'),
                         6*sum(response_tokens(r) for r in rows if r['dataset']=='limo'))
            initial=min(1_000_000,capacity); provisional=budget_plan(rows,initial)
            profile=self.main_timed('training',300*60,lambda:self.profile(rows,provisional))
            available=min(300*60-(time.time()-started),self.cutoff-time.time()-90*60)
            # Two independent runs; reserve model reload/save overhead at evaluation boundaries.
            estimate=2*len(provisional['groups'])*profile['seconds_per_update']*1.25+600
            target=int(initial*min(1.,max(0,available)/max(1,estimate)))
            budget={'target_tokens':target,'microbatch':profile['microbatch'],'form':form,
                    'profile':profile,'training_seconds_remaining_at_freeze':max(0,available),
                    'status':'ready' if target>=200_000 else 'insufficient_budget'}
            immutable_json(path,budget)
        self.summary['main_budget']=budget; self.save()
        if budget['status']!='ready': return
        plan=budget_plan(rows,budget['target_tokens'])
        self.summary['main_data']={k:v for k,v in plan.items() if k not in ('groups','cumulative_tokens')}
        self.summary['main_data']['limo_100_unique_reached']=plan['unique_by_source'].get('limo',0)>=100
        roots={}
        for name,lr in [('C_standard',5e-6),('C_gentle',1e-6)]:
            root=self.root/'training_trials'/name; roots[name]=root
            self.training_setup(root,rows,plan,lr,budget['microbatch'])
            self.summary.setdefault('training',{}).setdefault(name,{'status':'running','bad_streak':0})
        for step in sorted(set(plan['milestone_steps'].values())):
            for name,root in roots.items():
                record=self.summary['training'][name]
                if record['status'] not in ('running','complete'): continue
                if record.get('evaluated_step',0)>=step: continue
                try:
                    elapsed=self.main_timed('training',300*60,
                        lambda:self.train_until(root,step,f'{name}_train_to_{step}'))
                    record['train_wall_seconds']=record.get('train_wall_seconds',0)+elapsed
                    recipe={'training':str(root),'step':step,'alpha':1.,'kind':'main','form':form,'seed':42,
                            'supervised_tokens':plan['cumulative_tokens'][step-1]}
                    result=self.main_timed('evaluation',90*60,
                        lambda:self.evaluate(f'{name}_step{step}',self.adapter(root,step),recipe))
                    if result.get('status')=='complete':
                        metrics=result['metrics']; base=self.summary['baseline']
                        bad=metrics['correct']<=base['correct']-20 and metrics['correct_and_stop']<=base['correct_and_stop']
                        record['bad_streak']=record['bad_streak']+1 if bad else 0
                        record['evaluated_step']=step
                        if record['bad_streak']>=2: record.update(status='early_stopped',reason='two consecutive degraded milestones')
                    if step==len(plan['groups']) and record['status']=='running': record['status']='complete'
                except (PhaseTimeout,PhaseFailure) as exc:
                    record.update(status='time_limited' if isinstance(exc,PhaseTimeout) else 'failed',reason=str(exc))
                self.save()

    def confirm(self,best):
        candidate=self.summary['trials'][best]; recipe=candidate['recipe']
        original=Path(recipe['training']); rows=read(original/'pairs.json')
        plan=reshuffle_plan(rows,read(original/'schedule.json'),43)
        cfg=read(original/'context.json')['config']
        root=self.root/'training_trials/confirmation_seed43'
        self.training_setup(root,rows,plan,cfg['learning_rate'],cfg['microbatch'])
        self.train_until(root,recipe['step'],'confirmation_seed43_train')
        adapter=self.adapter(root,recipe['step'])
        if recipe['alpha']!=1.:
            adapter=self.interpolate('confirmation_seed43',adapter,recipe['alpha'])
        rep=self.evaluate('confirmation_seed43',adapter,dict(recipe,seed=43),selectable=False,slot='confirmation')
        expected=category(self.summary['baseline'],candidate['metrics'])
        def qualifies(base,result):
            kind=category(base,result['metrics']) if result.get('status')=='complete' else None
            return kind=='accuracy_preserving' or (expected=='termination_tradeoff' and kind=='termination_tradeoff')
        self.summary['replication']={'candidate':best,'seed43_passed':qualifies(self.summary['baseline'],rep),
            'replication_tokens_at_step':plan['cumulative_tokens'][recipe['step']-1],
            'protocol':'same selected multiset and full scheduler horizon; reshuffled seed43, same update checkpoint'}
        if not self.summary['replication']['seed43_passed']: return False
        baseline=self.repeat_dev(self.shared,'b0','b0_confirmation')
        # Evaluate the original selected weights again, in a fresh output directory.
        result=self.evaluate('selected_repeat',Path(candidate['adapter']),recipe,selectable=False,slot='confirmation')
        if result.get('status')!='complete': return False
        out=read(Path(result['path'])/'generations/rft_eval.json')
        fresh=candidate_report(baseline,out,self.dev)
        base=summarize(baseline,self.dev)
        passed=qualifies(base,dict(fresh,status='complete'))
        self.summary['repeat_dev']=dict(fresh,baseline=base,passed=passed)
        return passed

    def final_evaluation(self):
        selection=read(self.root/'selection.json'); self.summary['selection']=selection
        self.phase(self.shared/'context.json','generate',purpose='final',count=400,tag='final_b0')
        baseline=read(self.shared/'generations/b0_final.json')
        self.summary['final']={'baseline':summarize(baseline,self.final)}
        self.summary['final']['limo_baseline']=self.auxiliary(self.shared,'b0','test')
        half=self.old_plan['milestone_steps']['half']
        adapter=self.prior/f'training_trials/C_gsm_limo_nll/training/rft/milestones/step-{half:05d}'
        reference=self.evaluate('old_C_half',adapter,{'kind':'fixed_reference'},False,
                                purpose='final',slot='final_evaluations')
        self.summary['final']['fixed_C_half_reference']=reference
        self.summary.update(recommended_adapter=str(self.inputs/'stage2'),recommendation='keep_b0',
                            status='suite_complete_no_confirmed_improvement')
        if selection['candidate']:
            selected=self.summary['trials'][selection['candidate']]
            result=self.evaluate('selected',Path(selected['adapter']),selected['recipe'],False,
                                 purpose='final',slot='final_evaluations')
            self.summary['final']['candidate']=result
            if result.get('status')=='complete':
                intervals=result['outcome_intervals']; kind=result['category']
                self.summary['candidate_adapter']=selected['adapter']
                if kind and intervals['correct']['paired_bootstrap_95'][0]>0:
                    self.summary.update(status='suite_complete_accuracy_improvement',recommendation='accuracy_improvement_supported',
                                        recommended_adapter=selected['adapter'])
                elif kind and intervals['correct']['paired_bootstrap_95'][0]>-.03 and intervals['correct_and_stop']['paired_bootstrap_95'][0]>0:
                    self.summary.update(status='suite_complete_termination_tradeoff',recommendation='termination_tradeoff_supported',
                                        recommended_adapter=selected['adapter'])
                else:
                    self.summary['recommendation']='keep_b0_tradeoff_not_confirmed'
        incomplete={k:v.get('status') for k,v in self.summary.get('training',{}).items() if v.get('status')!='complete'}
        self.summary['incomplete_training']=incomplete
        self.summary['incomplete_evaluations']={k:v.get('status') for k,v in
            self.summary.get('final_evaluations',{}).items() if v.get('status')!='complete'}
        if self.summary['incomplete_evaluations']:
            self.summary['status']='suite_partial_final_evaluation'
        self.save()

    def run(self):
        self.summary['status']='running'
        self.stage('initialize',60,300)
        self.initialize()
        if not (self.root/'selection.json').exists():
            if not (self.root/'dev_winner.json').exists():
                self.stage('interpolation',90,300)
                try: self.interpolation_stage()
                except (PhaseTimeout,PhaseFailure) as exc:
                    self.summary['interpolation_stage_error']=str(exc); self.save()
                self.stage('format_controls',120,300)
                try: form=self.formats_stage()
                except (PhaseTimeout,PhaseFailure) as exc:
                    self.summary['format_stage_error']=str(exc); form='F00'
                    immutable_json(self.root/'format_choice.json',{'form':form,'reason':'time/failure fallback to raw + EOS','selected_before_main_training':True})
                self.stage('main_training_and_dev',390,300)
                try: self.main_stage(form)
                except (PhaseTimeout,PhaseFailure) as exc:
                    self.summary['main_stage_error']=str(exc); self.save()
                best=select(self.summary['baseline'],self.summary['trials'])
                immutable_json(self.root/'dev_winner.json',{'candidate':best,'selected_before_confirmation':True})
            best=read(self.root/'dev_winner.json')['candidate']
            selected=None
            if best:
                self.stage('confirmation',120,180)
                try:
                    if self.confirm(best): selected=best
                except (PhaseTimeout,PhaseFailure) as exc:
                    self.summary['confirmation_error']=str(exc)
            immutable_json(self.root/'selection.json',{'candidate':selected,'dev_winner':best,
                'selected_before_final':True,'policy':POLICY,
                'reason':'seed43 and repeated inference confirmed' if selected else 'no confirmed candidate'})
        # Reserve includes the 90-minute final and the 90-minute overrun allowance.
        self.cutoff=self.deadline
        self.final_evaluation()


def main():
    import re
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--source-suite',required=True,type=Path)
    parser.add_argument('--budget-minutes',type=int,default=960)
    parser.add_argument('--started-at',type=float)
    parser.add_argument('--train-python',default=sys.executable)
    parser.add_argument('--eval-python',required=True)
    parser.add_argument('--dtype',choices=['auto','float16','bfloat16'],default='auto')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id) or not 480<=args.budget_minutes<=960:
        parser.error('Use a safe run ID and a budget of 480–960 minutes')
    import torch
    if not torch.cuda.is_available(): parser.error('CUDA GPU required')
    if args.dtype=='auto': args.dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if args.dtype=='bfloat16' and not torch.cuda.is_bf16_supported(): parser.error('BF16 unsupported')
    args.workspace=args.workspace.expanduser().resolve(); args.workspace.mkdir(parents=True,exist_ok=True)
    with (args.workspace/'.math_training.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Another experiment owns this workspace')
        suite=CSuite(args)
        write(suite.root/'pid.json',{'pid':os.getpid(),'started_at':time.time()})
        def stop(*_): raise KeyboardInterrupt
        signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
        try: suite.run()
        except PhaseTimeout as exc: suite.summary.update(status='time_limited',error=str(exc))
        except KeyboardInterrupt:
            suite.terminate(); suite.summary['status']='interrupted'
        except Exception as exc:
            suite.summary.update(status='failed',error=str(exc)); raise
        finally:
            suite.summary['active_phase']=None; suite.save()
            write(suite.root/'status.json',{'status':suite.summary['status'],'ended_at':time.time()})
            print('STATUS:',suite.summary['status'],'ARTIFACTS:',suite.root,flush=True)


if __name__=='__main__': main()
