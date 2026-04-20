from collections import Counter, defaultdict
import heapq
import sys
import os
from time import sleep

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

	def minWindow(self, s, t):
		# 76 hard
		print(s,"\n" ,t)
		if t == "": return ""
		countT = {}
		window = {}

		for c in t:
			countT[c] = countT.get(c, 0) + 1
		have, need = 0, len(countT)
		res, resLen = [-1, -1], float("inf")
		l = 0

		for r in range(len(s)):
			c= s[r]
			window[c] = window.get(c, 0) + 1

			if c in countT and window[c] == countT[c]:
				have += 1
			while have == need:
				print_array_with_pointers(s, l, r)
			
				if (r-l+1) < resLen:
					res = [l, r]
					resLen = (r - l + 1)
				window[s[l]] -= 1
				if s[l] in countT and window[s[l]] < countT[s[l]]:
					have -= 1
				l += 1
		l,r = res
		print(l, r)
		return s[l:r+1] if resLen != float("inf") else ""

	def isValid(self, s):
		# 20 easy
		if len(s) % 2 == 1: return False
		stack = []
		closeSymbols = {
			  ")": "(",
			  "}": "{",
			  "]": "["
			  }
		length = 0
		for c in s:
			if c not in closeSymbols: # open symbol
				length+=1
				stack.append(c)
			else: # close symbol
				if length == 0: return False
				poped = stack.pop()
				length -= 1
				if closeSymbols[c] != poped:
					return False
		
		return length == 0  
				
	def findMin(self, nums):
		# 153 medium
		print(nums)
		l = 0
		r = len(nums) -1
		minimum = nums[0]
		while l <= r:
			print_array_with_pointers(nums, l, r)
			if nums[l] < nums[r]:
				minimum = min(minimum, nums[l])
				break
			mid = (l + r) //2
			minimum = min(minimum, nums[mid])
			if nums[mid] >= nums[l]:
				l = mid + 1
			else:
				r = mid - 1
		return minimum

	def search(self, nums, target):
		# 33 medium
		print(nums, target, "\n")
		l = 0
		r = len(nums) -1
		while l <= r:
			mid = (l + r) //2
			if nums[mid] == target:
				return mid			
			if nums[l] <= nums[mid]:
				if target > nums[mid]  or target < nums[l]:
					l = mid + 1
				else:
					r = mid - 1
			else: 
				if target < nums[mid] or target> nums[r]:
					r = mid - 1
				else: 
					l = mid + 1

	def reverseList(self, head):
		#206 easy
		showNodes(head)
		prev = None
		while head:
			t=head.next
			head.next = prev
			prev = head
			head = t
		showNodes(prev)
		return prev
		
	def mergeTwoLists(self, list1, list2):
		# 21 easy
		showNodes(list1)
		showNodes(list2)
		dumy = ListNode()
		tail = dumy
		while list1 and list2:
			if list1.val < list2.val:
				tail.next = list1
				list1 = list1.next
			else:
				tail.next = list2
				list2 = list2.next 
			tail = tail.next
		if list1:
			tail.next = list1
		elif list2:
			tail.next = list2

		showNodes(dumy)
		return dumy.next

	def hasCycle(self, head):
		# 141 easy
		print(head)
		slow = head
		fast = head

		while fast:
			slow = slow.next
			if fast.next :
				fast = fast.next.next 
			else: return False
			
			if slow == fast:
				return True
		return False

	def reorderList(self, head):
		# 143 medium
		showNodes(head)
		slow = head 
		fast = head.next
		
		while fast and fast.next:
			slow = slow.next
			fast = fast.next.next

		second = slow.next
		slow.next = None
		prev = None

		while second:
			tmp = second.next
			second.next = prev
			prev = second
			second = tmp
		first = head
		second = prev
		while second:
			tmp1 = first.next
			tmp2 = second.next
			first.next = second
			second.next = tmp1
			first = tmp1
			second = tmp2
		showNodes(head)
		
	def removeNthFromEnd(self, head, n): 	
		# 19 medium
		showNodes(head)
		print(n)

		# fast =head
		# slow = head

		# for i in range(n):
		# 	fast = fast.next
		# if fast is None:
		# 	return head.next
		# while fast.next is not None:
		# 	fast = fast.next
		# 	slow = slow.next
		# slow.next = slow.next.next

		# showNodes(head)
		# return head

		temp = head
		i=0
		prev = None
		while temp:
			i+=1
			tmp = temp.next
			temp.next = prev
			prev = temp
			temp = tmp
		
		temp = prev
		prev = None
		j = 0
		
		while temp:
			j+=1
			if j ==n:
				temp = temp.next
				if n == i:
					return prev
			print(temp)
			tmp = temp.next
			temp.next = prev
			prev = temp
			temp = tmp
		showNodes(prev)
		return prev

	def mergeKLists(self, lists):
		# 23 hard
		length = len(lists)
		if length == 0: return None
		
		def combineList(l1, l2):
			dumy = ListNode()
			tail = dumy
			while l1 and l2:		
				if l1.val < l2.val:
					tail.next = l1
					l1 = l1.next
				else:
					tail.next = l2
					l2 = l2.next
				tail = tail.next
			if l1:
				tail.next = l1
			elif l2:
				tail.next = l2
			return dumy.next
		
		while len(lists) > 1:
			mergedLists = []
			for i in range(0, len(lists), 2):
				l1 =lists[i]
				l2 = lists[i + 1] if i+1 < len(lists) else None
				mergedLists.append(combineList(l1, l2))
			lists = mergedLists
		
		return lists[0]
		






s = NeetCodeBlindQs()

test_arg1 =  [buildNodes([1,2,4]), buildNodes([1,3,5]), buildNodes([3,6])]
test_arg2 = 2
passes = test_arg1
out = leetcode_output(23, s.mergeKLists, passes)  #  // Output: [1,1,2,3,3,4,5,6]
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
# leetcode_output(76, s.minWindow, "OUZODYXAZV", "ZYX")  #  // Output: YXAZ
# leetcode_output(20, s.isValid, "((([][])))", )  #  // Output: true
# leetcode_output(153, s.findMin, [3,4,5,6,1,2] )  #  // Output: 1
# leetcode_output(33, s.search, [4,5,6,7,0,1,2], 0)  #  // Output: 4
# leetcode_output(206, s.reverseList, buildNodes([0,1,2,3]))  #  // Output: buildNodes([3,2,1,0])
# leetcode_output(21, s.mergeTwoLists, buildNodes([1,2,4]), buildNodes([1,3,5]))  #  // Output: buildNodes([1,1,2,3,4,5])
# leetcode_output(141, s.hasCycle, buildNodes([1,3,5]))  #  // Output: False // to make cycle use ListNode(0,next=node) & node.next = node2
# leetcode_output(143, s.reorderList, buildNodes([2,4,6,8,10]))  #  // Output: None 2-> 10-> 4-> 8-> 6-> -|
# leetcode_output(19, s.removeNthFromEnd, buildNodes([1,2,3,4]), 2)  #  // Output: [1,2,4]
# leetcode_output(23, s.mergeKLists, [buildNodes([1,2,4]), buildNodes([1,3,5]), buildNodes([3,6])])  #  // Output: [1,1,2,3,3,4,5,6]