from collections import Counter, defaultdict
import heapq
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from leetc.helpers import (
	ListNode,
	buildNodes,
	leetcode_output,
	print_array_with_pointers,
	print_matrix,
	showNodes,
)


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
		print(s, t)
		if len(s) != len(t):
			return False
		myMapS = {}
		myMapT = {}
		for i in range(len(s)):
			myMapS[s[i]] = 1 + myMapS.get(s[i], 0)
			myMapT[t[i]] = 1 + myMapT.get(t[i], 0)

		return myMapS == myMapT

	def twoSum(self, nums, target):
		# 1 easy
		print(nums)
		print(target)
		myMap = {}
		for i in range(len(nums)):
			curr = nums[i]
			diff = target - curr
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
		for n, c in myMap.items():
			freq[c].append(n)
		print(freq)
		output = []
		counter = 0
		for i in range(len(freq) - 1, 0, -1):
			for j in freq[i]:
				output.append(j)
				counter += 1
				if counter >= k:
					return output
		print(output, counter)

	def encode_decode(self, strs):
		# 271 medium
		def encode(self, strs):
			res = ""
			for s in strs:
				res += str(len(s)) + "#" + s
			return res

		def decode(self, s):
			print(s)
			res, i = [], 0
			while i < len(s):
				j = i
				while s[j] != "#":
					j += 1
				length = int(s[i:j])
				word = s[j + 1 : j + length + 1]
				res.append(word)
				i = j + 1 + length
			return res

		return decode(self, encode(self, strs))

	def productExceptSelf(self, nums):
		# 238 medium
		print(nums)
		leng = len(nums)
		res = [1] * leng
		# better solution
		pref = 1
		for i in range(leng):
			res[i] = pref
			pref *= nums[i]
			print(res, pref)
		postf = 1
		print("-----------")
		for i in range(leng - 1, -1, -1):
			res[i] *= postf
			postf *= nums[i]

		return res

	def longestConsecutive(self, nums):
		# 128 medium
		print(len(nums))
		longest = 0
		nSet = set(nums)
		for i in nSet:
			if (i - 1) not in nSet:
				length = 0
				j = i
				while (j + length) in nSet:
					length += 1
				longest = max(longest, length)
		return longest

	def isPalindrome(self, s):
		# 125 easy
		print(s)
		left, right = 0, len(s) - 1
		while left < right:
			if not s[left].isalnum():
				left += 1
				continue
			if not s[right].isalnum():
				right -= 1
				continue
			if s[left].lower() != s[right].lower():
				return False
			left += 1
			right -= 1
		return True

	def threeSum(self, nums):
		# 15 medium
		res = []
		nums.sort()
		print(nums)

		for i in range(len(nums)):
			if i > 0 and nums[i - 1] == nums[i]:
				continue
			print(nums[i])
			j = i + 1
			k = len(nums) - 1

			while j < k:
				sum = nums[i] + nums[j] + nums[k]
				if sum < 0:
					j += 1
				elif sum > 0:
					k -= 1
				else:
					res.append([nums[i], nums[j], nums[k]])
					j += 1
					while nums[j] == nums[j - 1] and j < k:
						j += 1

		return res

	def maxArea(self, heights):
		# 11 medium
		print(heights)
		r = len(heights) - 1
		l = 0
		maxx = 0
		while l < r:
			print(heights[l], heights[r])
			maxx = max(maxx, (r - l) * min(heights[l], heights[r]))
			if(heights[l] < heights[r]):
				l+=1
			else:
				r -= 1
		return maxx

	def maxProfit(self, prices):
		# 121 easy
		print(prices)
		maxxProfit = 0
		l, r = 0, 1
		while r < len(prices):
			if prices[l] < prices[r]:
				maxxProfit = max(maxxProfit, prices[r] - prices[l])
			if prices[r] < prices[l]:
				l=r
			r += 1
		return maxxProfit

	def lengthOfLongestSubstring(self, s):
		# 3 medium
		print(s)
		l= 0 
		res = 0
		charSet = set()
		for r in range(len(s)):
			while s[r] in charSet:
				charSet.remove(s[l])
				l += 1
			charSet.add(s[r])
			res = max(res, (r -l) + 1)
		return res

	def characterReplacement(self, s, k):
		# 424 medium
		print(s, k)
		count = {}
		res = 0
		l = 0
		maxF = 0

		for r in range(len(s)):
			print_array_with_pointers(s, l, r, (r -l + 1))
			count[s[r]] = 1 + count.get(s[r], 0)
			# maxF = max(count[s[r]], maxF)
			# while (r -l + 1) - maxF > k:
			while (r -l + 1) - max(count.values()) > k:
				print(f" updating .{(r -l + 1)}..." , count, max(count.values()))
				count[s[l]] -= 1
				l += 1
			res = max(res, r -l + 1)
		return res









s = NeetCodeBlindQs()

test_arg1 = "AAABABB"
test_arg2 = 1
passes = test_arg1
out = leetcode_output(424, s.characterReplacement, "AAABABB", 1)  #  // Output: 5
# print(out)


# leetcode_output( 217,s.hasDuplicate, [1,2,3,1]) #  // Output: True
# leetcode_output( 242,s.isAnagram, "racecar", "carrace") #  // Output: True
# leetcode_output( 1,s.twoSum, [2,7,11,15], 9) #  // Output: [0,1]
# leetcode_output( 49, s.groupAnagrams, ["eat","tea","tan","ate","nat","bat"]) # // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
# leetcode_output( 347, s.topKFrequent,[1,1,1,2,2,3], 2) # // Output: [1, 2]
# leetcode_output( 271, s.encode_decode, ["Hello","World"], ) #  // Output: ["Hello","World"]
# leetcode_output( 238, s.productExceptSelf, [1,2,3,4,]) #  // Output: [24,12,8,6]
# leetcode_output( 128, s.longestConsecutive, [100,4,200,1,3,2]) #  // Output: 4
# leetcode_output( 125, s.isPalindrome, "Was it a car or a cat I saw?") #  // Output: True
# leetcode_output( 15, s.threeSum,  [-1,0,1,2,-1,-4]) #  // Output: [[-1,-1,2],[-1,0,1]]
# leetcode_output(11, s.maxArea, [1, 7, 2, 5, 4, 7, 3, 6])  #  // Output: 36
# leetcode_output(121, s.maxProfit, [10,1,5,6,7,1])  #  // Output: 6
# leetcode_output(11, s.lengthOfLongestSubstring, "pwwkew")  #  // Output: 3
# leetcode_output(424, s.characterReplacement, "AAABABB", 1)  #  // Output: 5