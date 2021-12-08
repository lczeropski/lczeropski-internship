
#%%
from typing import Optional

#%%
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
#%%
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        p1 = head
        while p1 != None: 
            temp = p1.next
            p1.next = prev
            prev = p1
            p1 = temp
        print(self)
        return prev
# %%
# %%
#TODO: don't destroy head 
# %%
