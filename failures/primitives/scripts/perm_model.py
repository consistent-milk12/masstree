import itertools, random

def empty(W): return [W-1-i for i in range(W)], 0
def val(seq,size,W):
    v=size
    for i in range(W): v |= seq[i] << (4*(i+1))
    return v & ((1<<64)-1)
def unval(v,W):
    size=v&0xF
    seq=[(v>>(4*(i+1)))&0xF for i in range(W)]
    return seq,size

def insert_from_back(seq,size,i,W):
    assert i<=size and size<W
    slot=seq[W-1]
    seq=[x for x in seq]
    for p in range(W-1,i,-1): seq[p]=seq[p-1]
    seq[i]=slot
    return seq,size+1

def insert_back_formula(v,i,W):
    seq,size=unval(v,W)
    back=(v>>(4*W))&0xF
    i_shift=(i*4)+4
    low_mask=(1<<i_shift)-1
    nv=((v+1)&low_mask)|(back<<i_shift)|((v<<4)&~(low_mask|(0xF<<i_shift)))
    return nv & ((1<<64)-1), back

def remove_formula(v,i,W):
    seq,size=unval(v,W)
    if size==i+1:
        return (v-1)&((1<<64)-1)
    rot=(size-i-1)*4
    rot_mask=((((1<<rot)<<4)-1)<<((i+1)*4)) & ((1<<64)-1)
    nv=(((v-1)&~rot_mask)&((1<<64)-1))|(((v&rot_mask)>>4)&rot_mask)|(((v&rot_mask)<<rot)&rot_mask)
    return nv & ((1<<64)-1)

def remove_to_back_formula(v,i,W):
    mask=(~((1<<((i+1)*4))-1))&((1<<64)-1)
    shift_to_back=(W-i-1)*4
    x=v
    nv=(((x-1)&~mask)&((1<<64)-1))|((x>>4)&mask)|((x&mask)<<shift_to_back)
    return nv & ((1<<64)-1)

def remove_model(seq,size,i,W):
    seq=[x for x in seq]
    x=seq.pop(i); seq.insert(size-1,x)
    return seq,size-1
def remove_to_back_model(seq,size,i,W):
    seq=[x for x in seq]
    x=seq.pop(i); seq.insert(W-1,x)
    return seq,size-1

def make_sorted(n,W):
    seq=[i for i in range(n)]+[W-1-(p-n) for p in range(n,W)]
    return seq,n

for W in range(1,16):
    for v0,size0 in [(val(*empty(W),W),0)]:
        # exhaustive small sequences of ops for W<=4, random for larger
        pass
    ops_list=[('ins',i) for i in range(W)] + [('rem',i) for i in range(W)]
    # randomized
    for trial in range(20000):
        seq,size=empty(W)
        v=val(seq,size,W)
        ops=[]
        for _ in range(40):
            choices=['ins']*((W-size)>0)+['rem']*(size>0)+['remback']*(size>0)+['exch','swap']
            op=random.choice(choices)
            if op=='ins':
                i=random.randint(0,size)
                nv,slot=insert_back_formula(v,i,W)
                seq,size=insert_from_back(seq,size,i,W)
                v=val(seq,size,W)
                if nv!=v:
                    print("INSERT MISMATCH",W,trial,ops,i,nv,v,seq); raise SystemExit
            elif op=='rem':
                i=random.randint(0,size-1)
                nv=remove_formula(v,i,W)
                seq,size=remove_model(seq,size,i,W)
                v=val(seq,size,W)
                if nv!=v:
                    print("REMOVE MISMATCH",W,trial,ops,i,nv,v,seq); raise SystemExit
            elif op=='remback':
                i=random.randint(0,size-1)
                nv=remove_to_back_formula(v,i,W)
                seq,size=remove_to_back_model(seq,size,i,W)
                v=val(seq,size,W)
                if nv!=v:
                    print("REMBACK MISMATCH",W,trial,ops,i,nv,v,seq); raise SystemExit
            elif op=='exch':
                a=random.randint(0,W-1); b=random.randint(0,W-1)
                diff=((v>>((a+1)*4))^(v>>((b+1)*4)))&0xF
                nv=v^((diff<<((a+1)*4))|(diff<<((b+1)*4)))
                seq[a],seq[b]=seq[b],seq[a]
                v=val(seq,size,W)
                if nv!=v: print("EXCH MISMATCH",W,trial,a,b); raise SystemExit
            elif op=='swap':
                if size==W: continue
                a=random.randint(size,W-1); b=random.randint(size,W-1)
                if a==b: continue
                diff=((v>>((a+1)*4))^(v>>((b+1)*4)))&0xF
                nv=v^((diff<<((a+1)*4))|(diff<<((b+1)*4)))
                seq[a],seq[b]=seq[b],seq[a]
                v=val(seq,size,W)
                if nv!=v: print("SWAP MISMATCH",W,trial,a,b,seq,size); raise SystemExit
            ops.append(op)
    print("W",W,"ok")
