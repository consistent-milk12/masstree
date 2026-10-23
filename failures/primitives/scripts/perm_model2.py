import random
M=(1<<64)-1
def val(seq,size): 
    v=size
    for i,x in enumerate(seq): v|=x<<(4*(i+1))
    return v
def unval(v,W): return [(v>>(4*(i+1)))&0xF for i in range(W)], v&0xF
def logical(v,W): return unval(v,W)
def insert_model(seq,size,i,W):
    seq=list(seq); slot=seq[W-1]
    for p in range(W-1,i,-1): seq[p]=seq[p-1]
    seq[i]=slot; return seq,size+1
def remove_model(seq,size,i,W):
    seq=list(seq); x=seq.pop(i); seq.insert(size-1,x); return seq,size-1
def remback_model(seq,size,i,W):
    seq=list(seq); x=seq.pop(i); seq.insert(W-1,x); return seq,size-1
def insert_f(v,i,W):
    back=(v>>(4*W))&0xF; sh=(i*4)+4; lm=(1<<sh)-1
    return (((v+1)&lm)|(back<<sh)|((v<<4)&M&~(lm|(0xF<<sh))))&M
def remove_f(v,i,W):
    size=v&0xF
    if size==i+1: return (v-1)&M
    rot=(size-i-1)*4
    rm=((((1<<rot)<<4)-1)<<((i+1)*4))&M
    return ((((v-1)&M)&~rm)|(((v&rm)>>4)&rm)|(((v&rm)<<rot)&rm))&M
def remback_f(v,i,W):
    mask=(~((1<<((i+1)*4))-1))&M
    stb=(W-i-1)*4; x=v&M
    return (((x-1)&M&~mask)|((x>>4)&mask)|((x&mask)<<stb))&M
def exch_f(v,a,b):
    d=((v>>((a+1)*4))^(v>>((b+1)*4)))&0xF
    return (v^((d<<((a+1)*4))|(d<<((b+1)*4))))&M
def swap_f(v,a,b):
    d=((v>>((a+1)*4))^(v>>((b+1)*4)))&0xF
    return (v^((d<<((a+1)*4))|(d<<((b+1)*4))))&M
def rotate_f(v,i,j,W):
    if i==j or i==W: return v
    sh=(i+1)*4; mask=(1<<sh)-1
    x=v&M
    ra=(j-i)*4; rb=(W-j)*4
    return ((x&mask)|((x>>ra)&~mask&M)|((x&~mask&M)<<rb))&M
def rotate_model(seq,size,i,j,W):
    # hypothesis: positions i..j-1 move to back end (W-(j-i)..W-1), positions j..W-1 shift down
    seq=list(seq)
    mid=seq[i:j]; rest=seq[j:]
    new=seq[:i]+rest+mid
    # new length W; size unchanged
    return new,size
for W in [15]:
    for trial in range(3000):
        seq,size=unval(0,W) # empty initial
        seq=[W-1-p for p in range(W)]; v=val(seq,0)
        for step in range(60):
            choices=[]
            if size<W: choices+=['ins','ins','swap']
            if size>0: choices+=['rem','remback']
            choices+=['exch','rot']
            op=random.choice(choices)
            if op=='ins':
                i=random.randint(0,size); nv=insert_f(v,i,W); seq,size=insert_model(seq,size,i,W)
                exp=val(seq,size)
                if W==15 and nv!=exp: print("INS RAW",W,trial,step,nv,exp);raise SystemExit
                if logical(nv,W)!=logical(exp,W): print("INS LOG",W,trial,step,nv,exp);raise SystemExit
                v=nv
            elif op=='rem':
                i=random.randint(0,size-1); nv=remove_f(v,i,W); seq,size=remove_model(seq,size,i,W)
                exp=val(seq,size)
                if logical(nv,W)!=logical(exp,W): print("REM LOG",W,trial,step,i,nv,exp,seq,size);raise SystemExit
                v=nv
            elif op=='remback':
                i=random.randint(0,size-1); nv=remback_f(v,i,W); seq,size=remback_model(seq,size,i,W)
                exp=val(seq,size)
                if logical(nv,W)!=logical(exp,W): print("RB LOG",W,trial,step,i,nv,exp,seq,size);raise SystemExit
                v=nv
            elif op=='exch' or op=='swap':
                if op=='swap':
                    if size==W: continue
                    a=random.randint(size,W-1); b=random.randint(size,W-1)
                else:
                    a=random.randint(0,W-1); b=random.randint(0,W-1)
                nv=swap_f(v,a,b); seq[a],seq[b]=seq[b],seq[a]
                exp=val(seq,size)
                if logical(nv,W)!=logical(exp,W): print("SW",op,W,trial,step,a,b,nv,exp);raise SystemExit
                v=nv
            elif op=='rot':
                a=random.randint(0,W); b=random.randint(a,W)
                nv=rotate_f(v,a,b,W)
                seq2,size2=rotate_model(seq,size,a,b,W)
                if logical(nv,W)!=logical(val(seq2,size2),W):
                    print("ROT",W,trial,step,a,b,"seq",seq,"->",logical(nv,W),"expected",logical(val(seq2,size2),W));raise SystemExit
                seq,size=seq2,size2
                v=nv
    print("W",W,"ok")
