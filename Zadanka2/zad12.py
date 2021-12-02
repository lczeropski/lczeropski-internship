#%%
from typing import Optional


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
