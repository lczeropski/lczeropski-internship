#%%
from typing import Optional
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        s=head
        if(s):
            h=head.next
        while(s and h and h.next):
            if(s==h):
                return(True)
            s=s.next
            h=h.next.next
        return(False)
# %%
s = Solution
# %%
# %%
