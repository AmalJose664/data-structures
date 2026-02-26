from collections import Counter, defaultdict
import heapq
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from leetc.helpers import ListNode, buildNodes, leetcode_output, print_array_with_pointers, print_matrix, showNodes

class NeetCodeBlindQs(object):
	

	def hasDuplicate(self, nums):
		# 217 easy
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
		# 242 easy
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
		# 1 easy
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
		# 49 medium
		mymap = defaultdict(list)
		for s in strs:
			count = [0] * 26
			for c in s:
				count[ord(c) - ord("a")] += 1
			mymap[tuple(count)].append(s)
		return list(mymap.values())

	def topKFrequent(self, nums, k):
		# 347 medium
		myMap = {}
		freq = [[] for i in range(len(nums) + 1)]
		
		for n in nums:
			myMap[n] = myMap.get(n, 0) + 1
		for n,c in myMap.items():
			freq[c].append(n)
		print(freq)
		output = []
		counter = 0
		for i in range(len(freq)-1, 0 ,-1):
			for j in freq[i]:
				output.append(j)
				counter +=1
				if counter >= k:
					return output
		print(output, counter)
	





s = NeetCodeBlindQs()

test_arg1 =[1,2,2,2,3,3,4,4,4,4,5,5,5,5]
test_arg2 = 1
passes = test_arg2
leetcode_output( 347, s.topKFrequent, test_arg1, test_arg2) #  // Output: [1, 2]
# print(out)








# leetcode_output( 217,s.hasDuplicate, [1,2,3,1]) #  // Output: True
# leetcode_output( 242,s.isAnagram, "racecar", "carrace") #  // Output: True
# leetcode_output( 1,s.twoSum, [2,7,11,15], 9) #  // Output: [0,1]
# leetcode_output( 49, s.groupAnagrams, ["eat","tea","tan","ate","nat","bat"]) # // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
# leetcode_output( 347, s.topKFrequent,[1,1,1,2,2,3], 2) # // Output: [1, 2]
