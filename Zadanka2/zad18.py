class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        h = s = head
        for i in range(n):
            h = h.next
        if h is None:
            return head.next
        while h.next != None:
            h = h.next
            s = s.next
        s.next = s.next.next
        return head