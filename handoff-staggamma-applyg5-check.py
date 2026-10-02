# Replica of GridMilc StagGamma sign logic (one-link, U=1) on a 4^4 lattice.
import numpy as np, itertools
B={'G1':0,'GZ':1,'GY':2,'GYZ':3,'GX':4,'GZX':5,'GXY':6,'G5T':7,'GT':8,'GZT':9,'GYT':10,'G5X':11,'GXT':12,'G5Y':13,'G5Z':14,'G5':15}
GMU=[4,2,1,8]  # X,Y,Z,T -> grid dir 0..3
def lt(g):
    r=0;m=g
    for _ in range(3): m>>=1; r^=m
    return r
def gt(g):
    r=0;m=g
    for _ in range(3): m=(m<<1); r^=m
    return r&15
def par(b): return sum(1 for d in GMU if d&b)%2
def neg0(s,t):
    n=par(s&lt(s^t)); k=bin(s^t).count('1'); n^=(k*(k-1)//2)&1; return n
def mul(s1,t1,n1,s2,t2,n2):
    s,t=s1^s2,t1^t2; n=neg0(s,t)^(n1!=n2)^par((s1&lt(s2))^(t1&lt(t2))); return s,t,n
L=4; V=L**4
coords=np.array(list(itertools.product(range(L),repeat=4)))  # x,y,z,t
idx={tuple(c):i for i,c in enumerate(coords)}
def op(s,t,n):
    osc=lt(t)^gt(s); sh=s^t
    ph=np.array([(-1)**(n+sum(c[j] for j,d in enumerate(GMU) if d&osc)) for c in coords])
    M=np.zeros((V,V))
    if sh==0: M[range(V),range(V)]=1
    else:
        mu=[j for j,d in enumerate(GMU) if d&sh][0]
        for i,c in enumerate(coords):
            for dd in (1,-1):
                c2=c.copy(); c2[mu]=(c2[mu]+dd)%L; M[i,idx[tuple(c2)]]+=0.5
    return ph[:,None]*M
E=op(15,15,neg0(15,15))
for g in [('GX','G1'),('GY','G1'),('GZ','G1'),('G5X','G5'),('G5Y','G5'),('G5Z','G5')]:
    s,t=B[g[0]],B[g[1]]; n=neg0(s,t); G=op(s,t,n)
    s2,t2,n2=mul(s,t,n,15,15,neg0(15,15))
    kept=op(s2,t2,neg0(s2,t2))       # what ParseSpinTasteString yields (sign dropped)
    full=op(s2,t2,n2)                # algebraic product incl. sign
    def rel(A,Bm): return '+' if np.allclose(A,Bm) else ('-' if np.allclose(A,-Bm) else '?')
    herm = 'H' if np.allclose(G,G.T) else ('aH' if np.allclose(G,-G.T) else '?')
    hermk= 'H' if np.allclose(kept,kept.T) else ('aH' if np.allclose(kept,-kept.T) else '?')
    print(f"{g[0]:>4} {g[1]:<3} Γ:{herm:<2}  Γ̃:{hermk:<2}  kept vs Γ∘ε:{rel(kept,G@E)}  kept vs ε∘Γ:{rel(kept,E@G)}  full vs Γ∘ε:{rel(full,G@E)}")

print("--- all local + one-link pairs: eps∘Γ == (-1)^(neg(Γ) xor neg(P)) * O_P ? ; count where ParseSpinTasteString sign is wrong")
ok=0;bad=[];wrong=[]
names={v:k for k,v in B.items()}
for s in range(16):
  for t in range(16):
    if bin(s^t).count('1')>1: continue
    n=neg0(s,t); G=op(s,t,n); P=(s^15,t^15)
    sign=(-1)**(n^neg0(*P))
    if np.allclose(E@G, sign*op(*P,neg0(*P))): ok+=1
    else: bad.append((names[s],names[t]))
    if sign<0: wrong.append(f"({names[s]} {names[t]})")
print("formula holds:",ok,"bad:",bad)
print("pairs where current applyG5 gives -eps∘Γ:",len(wrong)); print(' '.join(wrong))
