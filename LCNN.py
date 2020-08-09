# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def findlayer(model,layernumber,wob):
# It is the function to find the index of the layer.
    if wob==0:
        layer='fc'+str(layernumber)+'.weight'
    else:
        layer='fc'+str(layernumber)+'.bias'
    i=0
    for name in model.state_dict():
        if name==layer:
            break
        i=i+1
        if name[:2]=='bn' and (name[4:]== 'running_mean'  or name[4:]== 'running_var' or name[4:]== 'num_batches_tracked'):
            i=i-1
    if i==len(model.state_dict()):
        raise IOError('Layer Name Error')
    return i


def BetaAdaptive(model,ln,i,beta=1.0):
# It is the function to do beta modificaiton for the i-th node in Layer ln.
    with torch.no_grad():
        firstweight=findlayer(model,ln,0)
        firstbias=findlayer(model,ln,1)
        secondweight=findlayer(model,ln+1,0)
        params = list(model.named_parameters())
        params[firstweight][1].data[i,:]=params[firstweight][1][i,:].data/beta
        params[firstbias][1].data[i]=params[firstbias][1].data[i]/beta
        params[secondweight][1].data[:,i]=params[secondweight][1].data[:,i]*beta

def adjust(model,images,threshold_u=100.0,threshold_l=0.1,scale=1.0,ln=1,oflag=0,shuff=0,mode=0):
    with torch.no_grad():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        L=len(images)
        hs=len(model.state_dict()['fc'+str(ln)+'.weight'])
        d_=torch.zeros(hs).to(device)
        for i in range(hs):
            d=0
            b=model.state_dict()['fc'+str(ln)+'.weight'][i].detach()
            a=model.state_dict()['fc'+str(ln)+'.bias'][i].detach()
            for k in range(L):
                image=images[k]
                l=-b/a
                x_k=1/image.to(device)
                d_k=(torch.abs(torch.sum(l*x_k)-1.0)/torch.norm(x_k))
                d=d+d_k

            d_[i]=d/L

        
        

        if mode==1:
            d_=d_**2

        if mode==2:
            for i in range(hs):
                a=torch.norm(model.state_dict()['fc'+str(ln)+'.weight'][i])
                b=torch.norm(model.state_dict()['fc'+str(ln)+'.bias'][i])
                c=torch.norm(model.state_dict()['fc'+str(ln+1)+'.weight'][:,i])
                
                det= torch.sqrt(c*c/len(model.state_dict()['fc'+str(ln+1)+'.weight'][:,i])/((b*b+a*a)/(len(model.state_dict()['fc'+str(ln)+'.weight'][i])+1)))
                if oflag>2:
                    print(det)
                
                d_[i]=d_[i]*d_[i]
                d_[i]=d_[i]/(det*det)


        d_=d_*scale

        d_=torch.min(d_,torch.tensor(threshold_u*1.0))
        d_=torch.max(d_,torch.tensor(threshold_l*1.0))


        ilist=np.arange(hs)




        if shuff>0:
            import random
            if shuff==1:
                random.shuffle(ilist)
            if shuff==2:
                for i in ilist:
                    d_[i]=random.uniform(0.1, 2)
                    
        for i in ilist:
            BetaAdaptive(model,ln,i,d_[i])

        if oflag>1:
            print(d_)

        if oflag>0:
            print ('Adjusting Layer {}, Kernel Nodes: {}, Adptive Nodes{}' .format(ln, torch.sum((d_<1).int()).item(), torch.sum((d_>1).int()).item()))


def accuracy(model,valloader):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    t=0
    c=0
    with torch.no_grad():
        for i, (images, labels) in enumerate(valloader):
            images = images.view(images.shape[0], -1).to(device)
            outputs = model(images)
            t=t+(torch.argmax(outputs,dim=1) == labels).float().sum()
            c=c+len(outputs)
    print('Accuracy: {:.4f} %' .format(t/c*100))
    return t/c*100

def one_hot(x, num_classes):

	return torch.eye(num_classes)[x,:]
