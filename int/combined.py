from collections import defaultdict
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
			
	def twoSum(self,nums,target):
		print(nums)
		print(target)
		myMap = {}
		for i in range(len(nums)):
			curr = nums[i]
			diff =  target - curr 
			if diff in myMap:
				return [myMap[diff], i]
			myMap[curr] = i
		return []

	def groupAnagrams(self, strs):
		# 49
		mymap = defaultdict(list)
		for s in strs:
			count = [0] * 26
			for c in s:
				count[ord(c) - ord("a")] += 1
			mymap[tuple(count)].append(s)
		return list(mymap.values())



s = NeetCodeBlindQs()

test_arg1 = ["eat","tea","tan","ate","nat","bat"]
test_arg2 = 9
passes = test_arg2
leetcode_output( 49, s.groupAnagrams, test_arg1) #  // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
# print(out)








# leetcode_output( 217,s.hasDuplicate, [1,2,3,1]) #  // Output: True
# leetcode_output( 242,s.isAnagram, "racecar", "carrace") #  // Output: True
# leetcode_output( 1,s.twoSum, [2,7,11,15], 9) #  // Output: [0,1]
# leetcode_output( 49, s.groupAnagrams, ["eat","tea","tan","ate","nat","bat"]) # // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
