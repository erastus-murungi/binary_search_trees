 def split(self,t,key):
       if t == None:
           return (None, None)
       if key < t.key:
           l,t.l = self.split(t.l,key)
           r = t
           return (l,r)
       else:
           t.r,r = self.split(t.r,key)
           l = t
           return (l,r)

 def merge(self,l,r):
       if l == None:
           return r
       elif r == None:
           return l
       elif l.priority > r.priority:
           l.r = self.merge(l.r,r)
           return l
       else:
           r.l = self.merge(l,r.l)
           return r
