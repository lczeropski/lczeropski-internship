class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        self.num = nums
        self.tgt = target
        for i in range(0,len(self.num)):
            for j in range(i+1,len(self.num)):
                if (self.num[i]+self.num[j])==self.tgt:
                    return [i,j]