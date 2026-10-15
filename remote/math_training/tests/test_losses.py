import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from losses import sequence_logps,dpo_loss,collate


def test_shift_mask_chunk_value_and_gradient():
    torch.manual_seed(42)
    x=torch.randn(2,7,11,requires_grad=True)
    y=torch.tensor([[-100,-100,3,4,5,6,7],[-100,1,3,4,-100,-100,-100]])
    sums,counts=sequence_logps(x,y,chunk=2)
    loss=-sums.sum()/counts.sum()
    expected=F.cross_entropy(x[:,:-1].reshape(-1,11),y[:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(loss,expected)
    a=torch.autograd.grad(loss,x,retain_graph=True)[0]
    b=torch.autograd.grad(expected,x)[0]
    torch.testing.assert_close(a,b)
    assert counts.tolist()==[5,3]


def test_dpo_reference_and_direction():
    c=torch.tensor([-4.],requires_grad=True)
    r=torch.tensor([-6.],requires_grad=True)
    loss=dpo_loss(c,r,c.detach(),r.detach(),.1).mean()
    torch.testing.assert_close(loss,torch.log(torch.tensor(2.)))
    loss.backward()
    assert c.grad.item()<0 and r.grad.item()>0
    assert dpo_loss(c.detach()+1,r.detach(),c.detach(),r.detach(),.1).item()<loss.item()


def test_padding_and_accumulation_invariance():
    torch.manual_seed(3)
    x=torch.randn(2,5,9,requires_grad=True)
    y=torch.tensor([[-100,2,3,4,5],[-100,-100,2,3,-100]])
    values,counts=sequence_logps(x,y)
    full=-values.sum()/counts.sum()
    split=sum(-sequence_logps(x[i:i+1],y[i:i+1])[0].sum()/counts.sum() for i in range(2))
    torch.testing.assert_close(full,split)
    padded_x=torch.cat([x,torch.randn(2,3,9)],dim=1)
    padded_y=torch.cat([y,torch.full((2,3),-100)],dim=1)
    torch.testing.assert_close(sequence_logps(padded_x,padded_y)[0],values)


def test_tiny_model_update_and_optimizer_resume(tmp_path):
    # Exercise real autograd/checkpointed likelihood, AdamW and state restoration on CPU.
    torch.manual_seed(7)
    model=torch.nn.Linear(5,13)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.02)
    inp=torch.randn(2,6,5)
    labels=torch.tensor([[-100,2,3,4,5,6],[-100,4,5,6,7,8]])
    refs=sequence_logps(model(inp),labels)[0].detach()
    def update(m,o):
        o.zero_grad();vals,_=sequence_logps(m(inp),labels,chunk=2)
        loss=dpo_loss(vals[:1],vals[1:],refs[:1],refs[1:],.1).sum()
        loss.backward();o.step()
    update(model,optimizer)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},tmp_path/'state.pt')
    update(model,optimizer)
    restored=torch.nn.Linear(5,13)
    other=torch.optim.AdamW(restored.parameters(),lr=.02)
    state=torch.load(tmp_path/'state.pt',weights_only=True)
    restored.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
    update(restored,other)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b)
