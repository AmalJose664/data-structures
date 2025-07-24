
import bisect
import collections
from itertools import groupby
import math
import sys
import os
import time

from helpers import ListNode, buildNodes, leetcode_output, print_array_with_pointers, showNodes
sys.path.append(os.path.dirname(__file__)) 

class MediumSolution(object):
    
	def addTwoNumbers(self, l1, l2):
		# 2
		dumy = ListNode()
		curr = dumy
		cary =0
		while l1 or l2 or cary:
			d1 = l1.val if l1 else 0 
			d2 = l2.val if l2 else 0

			sum = d1+d2+cary
			cary = sum //10
			print(sum)
			sum %=10
			print(sum ,cary)
			curr.next = ListNode(sum)

			curr = curr.next
			l1 = l1.next  if l1 else None
			l2 = l2.next  if l2 else None
		showNodes(dumy)
		return dumy.next
	
	def lengthOfLongestSubstring(self, s):
		# 3
		print(s ,)
		charSet = set()
		l=0
		res= 0

		for r in range(len(s)):
			print(f"====> checking '{s[r]}' with {s[l]}")
			while s[r] in charSet:
				print(f"{s[r]} in set")
				print(f"removing {s[l]} for {s[r]}")
				charSet.remove(s[l])
				l+=1
			charSet.add(s[r])
			res = max(res, r-l + 1)
			print(charSet,"----- res = ",res, f", for l = {l}, r = {r}, + = {r-l+1}")
		return res

	def longestPalindrome(self, s):
		# 5
		res= ""
		resLen = 0
		for i in range(len(s)):
			l = r = i
			while l>=0 and r <len(s) and s[l] == s[r]:
				print(f"for letter {s[i]}")
				if (r - l + 1) > resLen:
					res = s[l:r+1]
					resLen = r - l + 1
				l-=1
				r+=1
			l,r = i, i+1
			while l>=0 and r <len(s) and s[l] == s[r]:
				print(f"for letter {s[i]} 2nd")
				if (r - l + 1) > resLen:
					res = s[l:r+1]
					resLen = r - l + 1
				l-=1
				r+=1

		return res

	def convert(self, s, numRows):
		# 6
		if numRows == 1:
			return s
		ar = ['']*numRows
		print(s,numRows, ar, )
		j=0
		rev = False
		for i in range(1,len(s)+1):
			print(j, rev,i% numRows)
			ar[j] += s[i-1]
			if rev:
				j-=1
			else:
				j +=1
			
			if j == 0 or j == numRows -1:
				rev = not rev
		st = "".join(ar)
		return st

	def reverse(self, x):
		# 7
		MIN = -2**31
		MAX = (2**31)-1
		res = 0
		while x:
			digit = int(math.fmod(x, 10))
			x = int(x/10)

			if res>MAX//10 or (res ==MAX //10 and digit >= MAX % 10):
				return 0
			if res<MIN//10 or (res ==MIN //10 and digit <= MIN % 10):
				return 0
			res = (res * 10) + digit
		return res
	
	def myAtoi(self, s:str):
		# 8
		s = s.lstrip()		
		if not s:
			return 0
		number = 0
		sign = 1
		if s.startswith('-'):
			sign = -1
			s = s[1:]
		elif s.startswith('+'):
			s = s[1:]
		
		for i in range(len(s)):
			if not s[i].isdigit():
				break
			number = (10* number)+int(s[i])
		number = sign*number
		number = min(number, (2**31)-1)
		number = max(-(2**31), number)
		return number

	def maxArea(self, height):
		# 11
		l = 0
		r = len(height) - 1
		maxx = 0

		while l < r:
			minn = min(height[l], height[r])
			current = minn * (r - l)
			maxx = max(maxx, current)
			print(f"r-l = {r-l}, min = {minn}, current = {current}, r= {r}, l = {l}")
			while l < r and height[l] <= minn:
				l += 1 
				print(f"L reduce {l} now value {height[l]}")
			while l < r and height[r] <= minn:
				r -= 1
				print(f"r reduce {r} now value {height[r]}")
		

		return maxx
		
	def intToRoman(self, num):
		# 12
		roman_map = {
				1000: "M",
				900: "CM",
				500: "D",
				400: "CD",
				100: "C",
				90: "XC",
				50: "L",
				40: "XL",
				10: "X",
				9: "IX",
				5: "V",
				4: "IV",
				1: "I",
			}
		string = ""
		
		for key in roman_map:
			print(num, key, )
			while num>= key:
				string+=roman_map[key]
				num-=key
		# 	print(f"Usngi roman of {key} for num >>>>>>{num}")
		# 	if num //key:
		# 		count = num // key
		# 		print(f"count == {num//key} using key {key}")

		# 		string += (roman_map[key]*count)
		# 		print(string)
		# 		num = num % key
		# 		print(f" num % key ={num}")
		# print(string)
		return string

	def threeSum(self, nums):
		# 15
		res = []
		nums.sort()
		print(nums)
		for i, a in enumerate(nums):
			if i>0 and a ==nums[i-1]:
				continue
			l, r = i+1,len(nums)-1
			while l<r:
				sum = a  + nums[r] + nums[l]
				if sum>0:
					r-=1
				elif sum < 0:
					l+=1
				else:
					res.append([a, nums[l], nums[r]])
					l+=1
					while nums[l] == nums[l-1] and l<r:
						l+=1
			# if a>0:
			# 	break
		return res

	def threeSumClosest(self, nums, target):
		# 16
		if len(nums) == 3:
			return summ(nums)
		nums.sort()
		print(nums, target)
		res = float('inf')
		
		for i, a in enumerate(nums):
			if i>0 and a == nums[i-1]:
				continue
			l, r = i+1, len(nums) - 1
			while l<r:
				summ = a + nums[l] + nums[r]
				if abs(summ - target)< abs(res - target):
					res = summ

				if summ==target:
					return target
				elif summ< target:
					l+=1
				else:
					r-=1
		return res

	def letterCombinations(self, digits):
		# 17
		letters={
			'2': ['a', 'b', 'c'],
			'3': ['d', 'e', 'f'],
			'4': ['g', 'h', 'i'],
			'5': ['j', 'k', 'l'],
			'6': ['m', 'n', 'o'],
			'7': ['p', 'q', 'r', 's'],
			'8': ['t', 'u', 'v'],
			'9': ['w', 'x', 'y', 'z'],
		}
		l = len(digits)
		res = []
		def gets(i, curStr):
			if len(curStr) == l:
				res.append(curStr)
				return 
			for c in letters[digits[i]]:
				gets(i+1, curStr+c)
		if digits:
			gets(0,'')
		return res
		
	def fourSum(self,nums, target):
		# 18
		nums.sort()
		le = len(nums)
		print(nums, target)
		res, quad = [], []
		def kSum(k, start, target):
			if k!= 2:
				for i in range(start, le -k + 1):
					if i > start and nums[i] == nums[i-1]: # prevoius value equal or not
						continue
					quad.append(nums[i])
					print(f"fun start for {nums[i]}, target = {target}, k = {k}, to = {le-k+1}")
					kSum(k-1, i+1, target - nums[i])
					print(f"fun end for {nums[i]}")
					quad.pop()
					print(quad)
				return
			l ,r =start, le-1
			while l <r :
				print(f"While part l = {l}, r = {r}, target = {target}")
				if nums[l] + nums[r] < target:
					l+=1
				elif nums[l] + nums[r] > target:
					r-=1
				else:
					print(f"Adeed values {quad}, {nums[l]}, {nums[r]}")
					res.append(quad + [nums[l], nums[r]])
					l+=1
					while l< r and nums[l] == nums[l-1]:
						l+=1
		kSum(4, 0 ,target)
		print(res)

	def removeNthFromEnd(self, head, n):
		# 19
		if not head.next:
			return None
		print(n)
		showNodes(head)
		print()
		def show(node, prev):
			if not node:
				return [0,None] 
			ret = 1 + show(node.next, node)[0]
			if ret == n:
				if not prev:
					return [ret,node.next]
				else:
					prev.next = prev.next.next
			return [ret,node]
		t = show(head, None)
		print(t)
		showNodes(t[1])

	def generateParenthesis(self, n):
		# 22
		print(n)
		res = []
		paranthesis= []
		def backtrack(openB, closeB):
			if openB == closeB == n:
				res.append("".join(paranthesis))
				return
			if openB<n:
				paranthesis.append('(')
				backtrack(openB+1, closeB)
				paranthesis.pop()
			if closeB < openB:
				paranthesis.append(")")
				backtrack(openB, closeB+1)
				paranthesis.pop()
		backtrack(0,0)
		return res
			
	def swapPairs(self, head):
		# 24
		showNodes(head)
		dumy = ListNode(next=head)
		prev, curr = dumy, head

		while curr and curr.next:
			nxtPair  = curr.next.next
			second = curr.next
			print(curr.val, nxtPair, second, prev)

			second.next = curr
			curr.next = nxtPair
			prev.next = second

			prev = curr
			curr = nxtPair
		return showNodes(dumy.next)

	def divide(self, dividend,divisor):
		# 29
		print(dividend,divisor, )
		dvdt = abs(dividend)
		dvsr = abs(divisor)
		output = 0
		while dvdt>=dvsr:
			tmp = dvsr
			mul = 1
			while dvdt >= tmp:
				dvdt -= tmp
				output+=mul
				mul += mul
				tmp+=tmp
				print(f"output ={output}, mul = {mul}, tmp= {tmp}, {dvdt}")
			print(f"output ={output}, mul = {mul}, tmp= {tmp} --------")
		if (dividend<0) ^ (divisor<0):
			output= -output
		return min(2147483647, max(-2147483648, output))

	def nextPermutation(self, nums):
		# 31
		print(nums)

		def reverse_arr(start, end):
			while start< end:
				nums[start], nums[end] = nums[end], nums[start]
				start += 1
				end -= 1

		N = len(nums)-1
		le = N
		while le> -1:
			print("///////////",nums[le], le)
			if nums[le-1] < nums[le]:
				break
			le-=1
		print(le)
		if le <=0:
			reverse_arr(0,N)
			print(nums)
			return
		j = le-1
		print("Pick = ",nums[j])
		nxtMin = [nums[j], 0]
		for i in range(j+1, N+1):
			nxtMin = [nums[i],i] if nxtMin[0] < nums[i] or nums[j]<nums[i] else nxtMin
		print("To be changed = ",nxtMin[0])
		
		nums[j] ,nums[nxtMin[1]] = nxtMin[0], nums[j]
		
		print(nums,"---------------------")
		
		reverse_arr(j+1,N)
		print(nums)

	def search(self, nums, target):
		# 33
		print(nums)
		l,r = 0, len(nums)-1

		while l<= r:
			mid = (l+r)//2
			print(f"while loop mid = {mid}, left = {l}, right = {r}")
			if nums[mid] == target:
				return mid
			if nums[l] <= nums[mid]:
				print(" if part")
				if target > nums[mid] or target < nums[l]:
					print(f" grater {nums[mid]}")
					l = mid+1
				else:
					print(f" less {nums[mid]}")
					r=mid -1
			else:
				print(" ifelse part")
				if target< nums[mid] or target > nums[r]:
					print(f" grater {nums[mid]}")
					r = mid - 1
				else:
					print(f" grater {nums[mid]}")
					l = mid+1
		return -1

	def searchRange(self, nums, target):
		# 34
		def s(nums, target, leftB):
			l=0
			r = len(nums)-1
			print(nums)
			i=-1
			while l<=r:
				mid = (l+r)//2
				if nums[mid] == target:
					i=mid
					if leftB:
						r = mid-1	
					else:
						l = mid+1
				if target>nums[mid]:
					l = mid+1
				elif target<nums[mid]:
					r = mid-1
			return i
		return [s(nums,target, True),s(nums,target, False)]
		if not nums:return []
		l=0
		r = len(nums)-1
		print(nums)
		while l<=r:
			mid = (l+r)//2
			if nums[mid] == target:
				end = mid
				start = mid
				print(f"found at {mid}")
				while start > 0 and  nums[start-1] == target:
					start -=1
					print(f"reducing, {start}")
				while end< len(nums)-1 and nums[end+1] == target:
					end +=1
					print(f"increasing {end}")
				return [start,end]

			if target>nums[mid]:
				l = mid+1
			elif target<nums[mid]:
				r = mid-1
		return [-1, -1]

	def isValidSudoku(self, board):
		# 36
		# print sudoku 
		N=len(board)
		for i in range(N):
			if i % 3 == 0 and i != 0:
				print("-" * 21)  

			for j in range(N):
				if j % 3 == 0 and j != 0:
					print("|", end=" ") 

				print(board[i][j] if board[i][j] != "." else ".", end=" ")

			print()
		cols = collections.defaultdict(set)
		rows = collections.defaultdict(set)
		boxes = collections.defaultdict(set)
		for i in range(N):
			for j in range(N):
				value = board[i][j]
				if value == ".":
					continue
				if value in cols[j] or value in rows[i] or value in boxes[(i//3, j//3)]:
					return False
				cols[j].add(value)
				rows[i].add(value)
				boxes[(i//3, j//3)].add(value)
		
		return True

	def countAndSay(self, n):
		# 38
		def count(num):
			if not num:return[]
			arr = []
			times = 1
			current=num[0]
			for c in range(1,len(num)):
				current = num[c-1]
				if num[c] != current:
					arr.append([current,times])
					times = 1
				else:
					times+=1
			if num[-1] == current:
				arr.append([current,times])
			else:
				arr.append([num[-1],1])
			return arr
		
		def retrive2(arr):
			string = ""
			for i in arr:
				string += (str(i[1])+i[0])
			return string
		
		strs = "1"

		for i in range(n-1):
			strs = retrive2(count(strs))
		# print(retrive2(count(str(1))))
		return	strs

	def combinationSum(self, candidates, target):
		# 39
		print(candidates, target)
		res = []
		def find(i, cur, total):
			if total == target:
				res.append(cur.copy())
				return
			if i>= len(candidates) or total>target:
				return 
			
			cur.append(candidates[i])
			find(i,cur, total+candidates[i])
			cur.pop()
			find(i+1,cur, total)

		find(0, [], 0)
		print(res)
		return res

	def combinationSum2(self, candidates, target):
		# 40
		candidates.sort()
		res=[]
		def dfs(i, comb, total):
			if total == target:
				res.append(comb.copy())
				return
			
			if total> target or i==len(candidates):
				return
			
			comb.append(candidates[i])
			dfs(i+1, comb, total+candidates[i])
			comb.pop()
			
			while i+1<len(candidates) and candidates[i] == candidates[i+1]:
				i+=1
			
			dfs(i+1, comb, total)

		dfs(0,[],0)
		print(res)

	def multiply(self, num1, num2):
		# 43
		# if '0' in [num1, num2]:
			# return '0'
		n1, n2 = len(num1), len(num2)
		res = [0] * (n1 + n2)
		
		num1, num2 = num1[::-1], num2[::-1]
		print(num1)
		print(num2)
		for i1 in range(n1):
			for i2 in range(n2):
				digit = int(num1[i1]) * int(num2[i2])
				res[i1+i2] += digit
				res[i1 + i2 + 1] += (res[i1+i2] // 10)
				res[i1 + i2 ] = res[i1 + i2] % 10
				
		res, beg = res[::-1], 0
		while beg<len(res) and res[beg] == 0:
			beg+=1
		res = map(str, res[beg:])

		return "".join(res)

	def jump(self, nums):
		# 45
		print(nums)
		l =r =0
		res=0
		while r < len(nums)-1:
			farthest = 0
			for i in range(l, r+1):
				farthest = max(farthest, i+nums[i])
			l = r+1
			r = farthest
			res+=1
		return res

	def permute(self, nums):
		# 46
		print(nums)
		# iteration method
		# perms = [[]]

		# for n in nums:
		# 	print(f"for n {n}  ")
		# 	new_perm = []
		# 	print(f"  perms = {perms}")
		# 	for p in perms:
		# 		print(f"       p = {p}")
		# 		for i in range(len(p) + 1):
		# 			copy = p.copy()
		# 			copy.insert(i, n)
		# 			new_perm.append(copy)
		# 			print(f"       copy = {copy}")
		# 	perms = new_perm
		# return perms
		# backtrack method
		def backtrack(listt):
			
			if len(listt) ==0:
				return [[]]
			
			perms = backtrack(listt[1:])
			res = []
			for p in perms:
				for i in range(len(p) + 1):
					p_copy = p.copy()
					p_copy.insert(i, listt[0])
					res.append(p_copy)
			return res
		r = backtrack(nums)
		print()
		return r

	def permuteUnique(self, nums):
		# 47
		print(nums)
		res = []
		perms = []
		count = {}
		for n in nums:
			count[n] = count.get(n,0)+1
		print(count)
		def dfs():
			print("function call")
			print(perms)
			if len(perms) == len(nums):
				res.append(perms.copy())
				return
			for n in count:
				if count[n]>0:
					perms.append(n)
					count[n] -=1
					dfs()
					count[n] +=1
					perms.pop()
		dfs()
		return res
		
		sett = set()
		def backtrack(listt):
			if len(listt) == 0:
				return [[]]
			perms = backtrack(listt[1:])
			res =[ ]
			for p in perms:
				for i in range(len(p)+1):
					copy = p.copy()
					copy.insert(i, listt[0])
					tu = tuple(copy)
					if tu not in sett or True:
						res.append(copy)
						sett.add(tu)
			return res
		r = backtrack(nums)
		return r
	



s = MediumSolution()


# Problems till now 3600
test_arg1 = [1,1,2]
test_arg2 = [2,3,0,1,4]
passes =  test_arg1
leetcode_output( 46, s.permuteUnique, passes) #  // Output: [[1, 2, 3], [2, 1, 3], [2, 3, 1], [1, 3, 2], [3, 1, 2], [3, 2, 1]]
# print()





# leetcode_output( 2,s.addTwoNumbers, buildNodes([2,4,3]), buildNodes([5,6,4])) #  // Output: 7->0->8
# leetcode_output( 3,s.lengthOfLongestSubstring, "abcdabc",) #  // Output: 4
# leetcode_output( 5,s.longestPalindrome, 'babad') #  // Output: bab
# leetcode_output( 6,s.convert, 'PAYPALISHIRING' , 3) #  // Output: PAHNAPLSIIGYIR
# leetcode_output( 7, s.reverse, -123) #  // Output: -321
# leetcode_output( 8, s.myAtoi, " -098") #  // Output: -98
# leetcode_output( 11, s.maxArea, [1,8,6,2,5,4,8,3,7]) #  // Output: 49
# leetcode_output( 12, s.intToRoman, 3749) #  // Output: MMMDCCXLIX
# leetcode_output( 15, s.threeSum, [-1,0,1,2,-1,-4]) #  // Output: [[-1, -1, 2], [-1, 0, 1]]
# leetcode_output( 16, s.threeSumClosest, [-1,2,1,-4], 1) #  // Output: 2
# leetcode_output( 17, s.letterCombinations, '23') #  // Output: ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
# leetcode_output( 18, s.fourSum,[1,0,-1,0,-2,2], 0) #  // Output:  [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
# leetcode_output( 19, s.removeNthFromEnd,buildNodes([1,2,3,4,5,]), 2) #  // Output: [1,2,3,5]
# leetcode_output( 22, s.generateParenthesis,3) #  // Output: ['((()))', '(()())', '(())()', '()(())', '()()()']
# leetcode_output( 24, s.swapPairs , buildNodes([1,2,3,4])) #  // Output: [2,1,4,3]
# leetcode_output( 29, s.divide , 10, 3) #  // Output: 3
# leetcode_output( 31, s.nextPermutation, [1,4,3,2,6,5]) #  // Output:  [1, 4, 3, 5, 2, 6]
# leetcode_output( 33, s.search, [4,5,6,7,0,1,2], 0) #  // Output:  4
# leetcode_output( 34, s.searchRange, [8,8,8], 8) #  // Output:  [0,2]
# leetcode_output( 36, s.isValidSudoku, [[".",".",".",".","5",".",".","1","."],[".","4",".","3",".",".",".",".","."],[".",".",".",".",".","3",".",".","1"],["8",".",".",".",".",".",".","2","."],[".",".","2",".","7",".",".",".","."],[".","1","5",".",".",".",".",".","."],[".",".",".",".",".","2",".",".","."],[".","2",".","9",".",".",".",".","."],[".",".","4",".",".",".",".",".","."]]) #  // Output:  False
# leetcode_output( 34, s.countAndSay,4) #  // Output:  1211
# leetcode_output( 39, s.combinationSum,[2,3,5] , 8) #  // Output:  [[2,2,2,2],[2,3,3], [3,5]]
# leetcode_output( 40, s.combinationSum2,[10,1,2,7,6,1,5], 8) #  // Output: [[1,1,6],[1,2,5],[1,7],[2,6]]
# leetcode_output( 43, s.multiply,'10', '20') #  // Output: '200'
# leetcode_output( 45, s.jump,[2,3,1,1,4]) #  // Output: 2
# leetcode_output( 46, s.permute, [1,2,3]) #  // Output: [[1, 2, 3], [2, 1, 3], [2, 3, 1], [1, 3, 2], [3, 1, 2], [3, 2, 1]]