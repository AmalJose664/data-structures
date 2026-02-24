import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from leetc.helpers import ListNode, buildNodes, leetcode_output, print_array_with_pointers, print_matrix, showNodes

class NeetCodeBlindQs(object):
	

	def hasDuplicate(self, nums):
		# 217
		# easy answer
		# return len(set(nums)) != (len(nums))
		myMap = {}
		for i in nums:
			if i in myMap:
				return True
			else:
				myMap[i] = True
		return False
	
	def isAnagram(self, s, t):
		# 242
		print(s,t)
		if len(s) != len(t):
			return False
		myMapS = {}
		myMapT = {}
		for i in range(len(s)):
			myMapS[s[i]] = 1 + myMapS.get(s[i], 0)
			myMapT[t[i]] = 1 + myMapT.get(t[i], 0)
		
		return myMapS == myMapT
			
		
		
				




s = NeetCodeBlindQs()

test_arg1 = "résumés"
test_arg2 = "séruméa"
passes = test_arg2
leetcode_output( 242, s.isAnagram, test_arg1, passes) #  // Output: true
# print(out)








# leetcode_output( 217,s.hasDuplicate, [1,2,3,1]) #  // Output: True
# leetcode_output( 242,s.isAnagram, "racecar", "carrace") #  // Output: True
