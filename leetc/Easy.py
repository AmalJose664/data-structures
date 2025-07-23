from collections import deque
import sys
import os
import time
sys.path.append(os.path.dirname(__file__)) 

from helpers import Intervals, buildNodes, create_bst_tree, leetcode_output, print_tree_vertical, read4, showNodes, TreeNode, ListNode

class TwoSum():
	def __init__(self):
		self.ar=[]
		self.map = {}
	def add(self, number):
		self.ar.append(number)
		print(self.ar)
	def find(self,value):
		
		for n in self.ar:
			if value - n in self.map:
				return True
			self.map[n] = 1
		return False				

class MyStack(object):
	def __init__(self):
		self.q = deque()
	def push(self, x):
		self.q.append(x)
	def pop(self):
		for i in range(len(self.q) -1):
			self.push(self.q.popleft())
		return self.q.popleft()
	def top(self):
		return self.q[-1] if self.q else ''
	def empty(self):
		return len(self.q) == 0
	def __str__(self):
		return f"stack({str(self.q)})"

class MyQueue(object):
	def __init__(self):
		self.s1 = []
		self.s2 = []
	def push(self, x):
		self.s1.append(x)

	def pop(self):
		if not self.s2:
			while self.s1:
				self.s2.append(self.s1.pop())
		return self.s2.pop()	

	def peek(self):
		if not self.s2:
			while self.s1:
				self.s2.append(self.s1.pop())
		return self.s2[-1]

	def empty(self):
		return max(len(self.s1), len(self.s2)) == 0

class EasySolution(object):
	def twoSum(self,nums,target):
		# 1
		map = {}
		for i in range(len(nums)):	
			if(target - nums[i] in map):
				return [map[target - nums[i]], i]
			map[nums[i]] = i
		return []
        
	def isPalindrome(self, x):
		# 9
		if(x<0): return False

		rev = 0
		num = x
		while(num > 0):
			
			rev = (rev * 10) + num % 10
			num = num // 10

		return rev == x
	
	def romanToInt(self, s):
		# 13
		romanMap={
			"I": 1,
			"V": 5,
			"X": 10,
			"L": 50,
			"C": 100,
			"D": 500,
			"M": 1000
		}
		number = 0
		prev = 0
		for char in reversed(s):
			curr = romanMap[char]
			
			if(curr < prev):
				number-=curr
			else:
				number+=curr
			prev = curr

		return number
	
	def longestCommonPrefix(self, strs):
		# 14

		# prefix=''
		# small_word_len = len(strs[0])
		# small_word = strs[0]
		# for i in range(len(strs) -1):
		# 	if( len(strs[i+1]) < small_word_len):
		# 		small_word = strs[i+1]
		# 		small_word_len = len(strs[i+1])



		# for i in range(len(small_word)):
		# 	for str in strs:
		# 		if(small_word[i] != str[i]):
		# 			return prefix
				
		# 	prefix += small_word[i]
		# return prefix
		strs.sort()
		first = strs[0]
		last = strs[-1]
		result = ''
		for i in range(len(first)):
			if(i == len(last) or first[i] != last[i]):
				return result
			result += first[i]
		return result
	
	def isValid(self, s):
		# 20
		if(len(s) % 2 ==1):return False
		stack = []	
		map = {")":"(","]":"[","}":"{"}
		for br in s:
			if(br in map):
				if(stack and stack[-1] == map[br]):
					stack.pop()
				else:
					return False
			else:
				stack.append(br)
			

		return True if not stack else False
	
	def mergeTwoLists(self, list1, list2):
		# 21
		dummy = ListNode()
		curr = dummy

		while list1 and list2:
			if list1.val > list2.val:
				curr.next = list2
				curr = list2 
				list2 = list2.next
			else:
				curr.next = list1
				curr = list1
				list1 = list1.next
				
		curr.next = list1 if list1 else list2
		showNodes(dummy.next)
		return dummy.next 

	def removeDuplicates(self, nums):
		# 26
		j=1
		for i in range(1,len(nums)):
			if(nums[i] != nums[i-1]):
				nums[j] = nums[i]
				j=j+1
		print(nums)
		return j

	def removeElement(self, nums, val):
		# 27
		insertPtr=len(nums) -1
		searchPtr = 0

		while insertPtr > searchPtr:
			
			if nums[insertPtr] == val:
				insertPtr-=1
			elif nums[searchPtr] == val:
				nums[searchPtr] = nums[insertPtr]
				nums[insertPtr] = val
				searchPtr+=1
				insertPtr-=1
			else:
				searchPtr+=1
		searchPtr=0
		for n in nums:
			if( n == val):
				break
			searchPtr+=1
		return searchPtr
		# ANother simple one 
		i=0
		for j in range(len(arr)):
			if(arr[j] != val):
				temp = arr[j]
				arr[j] = arr[i]
				arr[i] = temp
				i+=1
		print(arr)
	
	def strStr(self, haystack, needle):
		# 28
		c= needle[0]
		for ch in range(len(haystack)):
			if(haystack[ch] ==c):
				if(haystack[ch:ch+len(needle)] == needle):
					return ch
		return -1	
				
	def searchInsert(self, nums, target):
		# 35
		n = len(nums)
		l = 0
		r = n-1

		while(l<=r):
			m = (l+r) // 2
			if nums[m] < target:
				l = 1 + m
			elif nums[m] > target:
				r = m - 1
			else:
				return m 
		if(nums[m] < target):
			return m +1
		else:
			return m

	def lengthOfLastWord(self, s):
		# 58
		end = len(s) -1
		while s[end] == " ":
			end-=1
		start = end
		while start>=0 and s[start] != " ":
			start-=1
		return end-start

	def plusOne(self, digits):
		# 66

		# newArr = []
		# number = 0
		# for i in range(len(digits)):
		# 	number = number *10 + digits[i]
		# number = number + 1
		
		# while (number > 0):
		# 	temp = number
		# 	number = number // 10
		# 	newArr.insert(0,temp % 10)
		# return newArr
		
		for i in range(len(digits) -1, -1, -1):
			if digits[i] < 9:
				digits[i] += 1
				return digits
			else:
				digits[i] = 0
		return [1]+ digits
	
	def addBinary(self, a, b):
		# 67
		a,b = int(a,2) ,int(b,2)
		while b:
			without_cry = a ^ b
			carry = (a & b) << 1
			a, b = without_cry, carry
		return bin(a)[2:]
		'''
			bin(a+b)[2:] // simple
		'''

	def mySqrt(self, x):
		# 69
		left = 1
		right = x
		while(left <= right):
			mid = (left + right) //2
			if mid*mid == x:
				return mid
			elif mid*mid < x:
				left = mid +1
			else:
				right = mid -1
		return right
	
	def climbStairs(self, n):
		# 70
		if(n == 1):return 1
		if(n == 2):return 2
		sum = 2
		prev =1
		i=2
		while i < n:
			prev , sum = sum, prev + sum
			i+=1
		return sum
		# dynamic programing way 
		memo = {1:1, 2:2}
		def f(n):
			if n in memo:
				return memo[n]
			else:
				memo[n] = f(n -2) + f(n-1)
				return memo[n]
		return f(n)

	def deleteDuplicates(self, head):
		# 83
		temp = head
		while temp and temp.next:
			if temp.val == temp.next.val:
				temp.next = temp.next.next
			else:
				temp = temp.next
		return head

	def merge(self, nums1, m, nums2, n):
		# 88
		y=n -1
		x=m -1
		for z in range(m+n-1, -1, -1):
			if x < 0:
				nums1[z] = nums2[y]
				y-=1
			elif y < 0:
				break
			elif nums1[x]> nums2[y]:
				nums1[z] = nums1[x]
				x-=1
			else:
				nums1[z] = nums2[y]
				y-=1
		return nums1
					
	def inorderTraversal(self, root):
		# 94
		result = []
		def traverse(node):
			if not node:
				return result			
			traverse(node.left)
			result.append(node.val)
			traverse(node.right)
		traverse(root)
		return result

	def isSameTree(self, p, q):
		# 100
		def isSame(a,b):
			if not a and not b:
				return True
			if (a and not b) or (b and not a):
				return False
			if a.val != b.val:
				return False
			return isSame(a.left, b.left) and isSame(a.right,b.right)
				
		return isSame(p,q)
	
	def isSymmetric(self, root):
		# 101
		def same(root1,root2):
			if not root1 and not root2:
				return True
			if not root1 or not root2:
				return False
			if root1.val != root2.val:
				return False
			return same(root1.left, root2.right) and same(root1.right, root2.left)

		return same(root,root)

	def maxDepth(self, root):
		# 104
		def count(node):
			if not node:
				return 0
			lenL=count(node.left)
			lenR =count(node.right)
			print(node.val, lenL, lenR)
			return max(lenL,lenR) + 1

		return count(root)
	
	def sortedArrayToBST(self, nums):
		# 108
		def create(nums, left , right):
			if(left>right):
				return None
			mid = left+ (right-left) //2
			t = TreeNode(nums[mid])
			t.left = create(nums, left, mid-1)
			t.right = create(nums, mid+1, right)
			return t
		return create(nums, 0 , len(nums)-1)
			
	def isBalanced(self, root):
		# 110
		balanced = [True]
		def height(node):
			if not node:
				return 0
			left = height(node.left)
			if balanced[0] is False:
				return 0
			
			right = height(node.right)
			print(f"Left = {left} Right = {right} for value {node.val} minus = {left-right}")
			if abs(left - right) >1:
				balanced[0] = False
				return 0
			
			return 1 + max(left , right)
		height(root)
		return balanced[0]

	def minDepth(self, root):
		# 111
		if not root:
			return 0
		if not root.left:
			return self.minDepth(root.right) + 1
		if not root.right:
			return self.minDepth(root.left) + 1 
		
		return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
		
	def hasPathSum(self, root, targetSum):
		# 112
		print_tree_vertical(root)
		def find(node,val):
			if not node:
				return False
			
			val+=node.val
			if val == targetSum and (not node.left and not node.right):
				return True
			return find(node.left,val) or find(node.right,val)
		
		return find(root, 0)
			
	def generate(self, numRows):
		# 118
		rows = [[1]]
		for _ in range(numRows-1):
			temp = [0]+rows[-1]+[0]
			row=[]
			for j in range(len(rows[-1])+1):
				row.append(temp[j] + temp[j+1])
			rows.append(row)
		return rows

	def getRow(self, rowIndex):
		# 119
		last = [1]
		for _ in range(rowIndex):
			temp = [0]+last+[0]
			row=[]
			for j in range(len(last)+1):
				row.append(temp[j] + temp[j+1])
			last = row
		return row

	def maxProfit(self, prices):
		# 121
		minV = prices[0]
		maxP = 0
		for price in prices:
			minV = price if price < minV else minV
			maxP = maxP if price - minV < maxP else price - minV

		return maxP

	def validPalindrome(self, s=""):
		# 125
		cleaned = "".join(c.lower() for c in s if c.isalnum())
		return cleaned == cleaned[::-1]
		
	def singleNumber(self, nums):
		# 136
		x=0
		for n in nums:
			x^=n
		return x
	
	def hasCycle(self, head):
		# 141

		# fast = head
		# slow = head
		# while fast and fast.next:
		# 	print(f"Fast = {fast.val} slow = {slow.val}")
		# 	slow= slow.next
		# 	fast = fast.next.next
		# 	if slow == fast:
		# 		return True
		# return False

		# me code dowm
		if not head :return False
		map={}
		while not head in map:
			print(head.val)
			if not head.next:
				return False
			map[head] = head.val
			head = head.next
		return True

	def preorderTraversal(self, root):
		# 144
		nodes = []
		print_tree_vertical(root)
		def traverse(node):
			if not node:
				return
			nodes.append(node.val)
			traverse(node.left)
			traverse(node.right)
		traverse(root)
		return nodes
	
	def postorderTraversal(self, root):
		# 145
		nodes = []
		print_tree_vertical(root)
		def traverse(node):
			if not node:
				return
			traverse(node.left)
			traverse(node.right)
			nodes.append(node.val)
		traverse(root)
		return nodes
	
	def read(self, buf, n):
		# 157 
		r= 0
		buf4=[None]*4
		while r < n:
			L = read4(buf4)
			print(L, buf4)
			r+=L
			if L ==0:
				break
			buf[r-L:r] = buf4[:]
		return min (r,n)
	
	def getIntersectionNode(self, headA, headB):
		# 160
		showNodes(headA)
		showNodes(headB)
		
		l1 = headA
		l2 = headB
		while l1 != l2:
			print(f"loop work values l1 = {l1.val if l1 else '--'},    l2 = {l2.val if l2 else '--'}")
			l1 = l1.next if l1 else headB
			l2 = l2.next if l2 else headA
		return l1
			
	def findMissingRanges(self, nums, lower, upper):
		# 163
		n = len(nums)
		print(nums)
		missing_ranges = []
		if n ==0:
			return [[lower, upper]]
		if lower<nums[0]:
			missing_ranges.append([lower, nums[0]-1])
		for i in range(n-1):
			if nums[i+1] - nums[i] <=1:
				continue
			print("For i, ",i)
			missing_ranges.append([nums[i]+1, nums[i+1]-1])
		if upper > nums[n-1]:
			missing_ranges.append([nums[n-1], upper])
		return missing_ranges
		
	def convertToTitle(self, columnNumber):
		#  168
		str=''
		while columnNumber>0:
			offset = (columnNumber - 1) % 26
			str= chr(ord('A')+ offset) + str
			columnNumber = (columnNumber - 1)// 26
		return str
	
	def majorityElement(self, nums):
		# 169
		count=0
		candidate=0
		print(nums)
		for n in nums:
			if count == 0:
				candidate = n
			count += 1 if candidate ==n else -1
			 
		return candidate
			
	def twoSum3(self, nums, target):
		# 170
		two = TwoSum()
		for n in nums:
			two.add(n)
		return two.find(target)
		
	def titleToNumber(self, columnTitle):
		# 171
		number=0
		for c in columnTitle:
			number = number*26 + ord(c) -64
		return number

	''' 175 combine two table => SQL find in site
		select Person.firstName, Person.lastName, Address.city, Address.state from Person left join Address on Person.personId = Address.personId
	'''
	
	''' 181 Find from same table => SQL find in site
		select e.name as employee from Employee e join Employee m on e.managerId = m.id where e.salary > m.salary
	'''
	
	''' 182 Find duplicate emails
		select email from Person group by email having count(email)>1  '''
	
	''' 183 Select customers who are not ordered yet
		select name as Customers from  Customers where id not in (select customerId from Orders)	
	'''
	
	def reverseBits(self,n):
		# 190
		# return int(bin(n)[2:].zfill(32)[::-1],2 )
		# bit manupulation
		res= 0
		print(n)
		for i in range(32):
			bit = (n >> i) & 1
			res = res | (bit << (31 - i))
		return res

	def hammingWeight(self, n):
		# 191
		# return bin(n)[2:].count('1')
		res= 0 
		while n:
			# print(f"loop work, {bin(n)[2:]}, {bin(n-1)[2:]}")
			# res+=n % 2
			# n=n>>1
			n=n&(n-1)
			res+=1
		print(res)

	r''' 193 Valid phone numbers
		grep -e "^[0-9]\{3\}\-[0-9]\{3\}\-[0-9]\{4\}$" -e "^([0-9]\{3\}) [0-9]\{3\}\-[0-9]\{4\}$" file.txt
	
	'''	

	''' 195 Tenth Line
		awk "NR==10" file.txt
	'''

	''' 196 Delete Duplicate
		delete p1 from Person p1, Person p2 where p2.id<p1.id and p1.email = p2.email
	'''

	''' 197 Rising Temperature
		select w2.id from Weather w1 join Weather w2 on w2.recordDate = DATE_ADD(w1.recordDate, INTERVAL_1_DAY) where w2.temperature > w1.temperature
	'''

	def isHappy(self, n):
		# 202
		st = set()
		sum=0
		while n not in st:
			st.add(sum)
			sum=0
			while n>0:
				sum+= (n%10)**2
				n=n//10
			if sum ==1:
				return True	
			n=sum
		return False

	def removeElements(self,head, val):
		# 203
		showNodes(head)
		curr = head
		dumy = ListNode(next=head)
		prev = dumy
		while curr:
			if curr.val == val:
				prev.next = curr.next
			else:
				prev = prev.next
			curr = curr.next
		return dumy.next
		showNodes(dumy.next)
	
	def isIsomorphic(self, s,t):
		# 205
		mapSt, mapTs={},{}
		for i in range(len(s)):
			c1,c2 = s[i],t[i]
			if (c1 in mapSt and mapSt[c1] != c2) or (c2 in mapTs and mapTs[c2] != c1):
				return False
			mapSt[c1] = c2
			mapTs[c2] = c1
		return True
	
	def reverseList(self, head):
		# 206
		showNodes(head)
		prev = None
		while head:
			t =head.next
			head.next = prev
			prev = head
			head = t
		showNodes(prev)
		return prev
	
	def containsDuplicate(self, nums):
		# 217
		map={}
		for n in nums:
			if n in map:
				return True
			map[n] = n
		return False

	def containsNearbyDuplicate(self, nums,k):
		# 219
		window = set()
		L = 0
		for R in range(len(nums)):
			print(f"r-l = {R-L}, R = {R}, L={L}, window ={window} cuurent value {nums[R]}\n")
			if R - L > k:
				window.remove(nums[L])
				print(f"Increasing L, window removed, => {window}")
				L+=1
			if nums[R] in window:
				return True
			window.add(nums[R])
		return False

	def countNodes(self, root):
		print_tree_vertical(root)
		def left(node):
			if not node: return 0
			print(f"calls left  current = {node.val} from left !!!")
			return left(node.left) + 1
		def right(node):
			if not node: return 0
			print(f"calls right current = {node.val} from right!!")
			return right(node.right) + 1
		def add(node):
			if not node: return 0
			l_height = left(node)
			r_height = right(node)
			print(f"calls main!! left = {l_height} right = {r_height}")
			if l_height == r_height:
				return (2**l_height) -1
			else:
				return add(node.left) + add(node.right) + 1	
		return add(root)
		
		# def count(node):
		# 	if not node:
		# 		return 0
		# 	return count(node.left) + count(node.right) + 1
		# return count(root)

	def stackUsingQues(self,):
		# 225
		# class code on top of file
		st = MyStack()
		st.push(2)
		st.push(3)
		print(st)
		print(st.pop())
		print(st.top())
		print(st.empty())

	def invertTree(self, root):
		# 226
		# print_tree_vertical(root)
		if not root:
			return 
		
		self.invertTree(root.left)
		self.invertTree(root.right)
		temp = root.left
		root.left = root.right
		root.right = temp
		
		return root

	def summaryRanges(self, nums):
		# 228
		ar=[]
		length = len(nums)
		start = nums[0]
		print(nums)
		for i in range(1, length+1):
			if i == length or nums[i] != nums[i-1]+1:
				if start == nums[i-1]:
					ar.append(f"{start}")
				else:
					ar.append(f"{start}->{nums[i-1]}")
				if i<length:
					start = nums[i]
		return ar
		# another answer
		i=0
		while i < length:
			start = nums[i]
			while i<length-1 and nums[i] +1 == nums[i+1]:
				i+=1
			if start != nums[i]:
				ar.append(str(start)+"->"+str(nums[i]))
			else:
				ar.append(str(start))
			i+=1
		
		return ar

	def isPowerOfTwo(self, n):
		# 231
		t=1
		if n == 1:
			return True
		while t<=n:
			t *= 2 
			if t == n:
				return True
		return False

	def queUsingStacks(self):
		# 232
		q = MyQueue()
		q.push(4)
		q.push(8)
		q.push(16)
		q.push(32)
		print(q.pop())
		print(q.pop())
		print(q.peek())
		print(q.empty())

	def isPalindromeLinked(self, head):
		# 234
		fast = head
		slow = head
		while fast and fast.next:
			slow = slow.next
			fast = fast.next.next
		prev = None
		while slow:
			temp = slow.next
			slow.next = prev
			prev = slow
			slow = temp
		
		left, right = head, prev
		while right:
			if left.val != right.val:
				return False
			left,right = left.next, right.next
		return True

	def isAnagram(self, s,t):
		# 242
		if len(s) != len(t):
			return False
		map1 = {}
		map2 = {}
		for i in  range(len(s)):
			map1[s[i]] = 1 +map1.get(s[i],0)
			map2[t[i]] = 1 + map2.get(t[i],0)
		
		for m in map1:
			if map1[m] != map2.get(m,0):
				return False
		return True

	def shortestDistance(self, words, word1, word2):
		# 243
		index1 = index2 = -1
		min_dist = float('inf')
		for i, word in enumerate(words):
			if word == word1 == word2:
				if index1 != -1:
					min_dist = min(min_dist, i- index1)
				index1 = i
			elif word  == word1:
				if index2 != -1:
					min_dist = min(min_dist, i- index2)
				index1 = i
			elif word == word2:
				if index1 != -1:
					min_dist = min(min_dist, i- index1)
				index2 = i
		return min_dist

	def isStrobogrammatic(self, num):
		# 246 # 644
		n = int(num)
		map={
		"6":'9',
	   '9':'6',
	   '8':'8',
	   '0':'0',
	   '1':'1'
	   }
		new = ''
		for s in num:
			if s in map:
				new+=map[s]
			else:
				return False
		print(f"original = {n},\nnew = {new}, \nnewflipped = {new[::-1]}")
		return n == int(new[::-1])

	def can_attend_meetings(self, intervals=[]):
		# 252 # 920
		intervals.sort(key = lambda i : i.start)
		for n in range(1,len(intervals)):
			if intervals[n-1].end >intervals[n].start:
				return False
		return True
			
	def binaryTreePaths(self, root):
		# 257
		print_tree_vertical(root)
		
		def find(node, path):
			if not node:
				return 
			path+=f"{node.val}"
			if not node.left and not node.right:
				ans.append(path)
				return
			else:
				find(node.left, path+"->")
				find(node.right, path+'->')
		ans = []
		find(root,'')
		return ans
		
	def addDigits(self, num):
		# 258 easy way 
		if num<10:
			return num
		else:
			return ((num -1) % 9) +1
		# hard way
		while True:
			sum = 0
			while num>0:
				
				sum += num % 10
				print('sum = ',sum)
				num = num // 10
			num = sum
			if num>=0 and num<10:
				return num
		return num

	def isUgly(self, n):
		# 263
		if n<=0:
			return False
		for p in [2,3,5]:
			while n%p == 0:
				n = n//p
			return n == 1
		
	def can_permute_palindrome(self, s):
		# 266 916
		map = {}
		for c in s:
			if c not in map: 
				map[c]=1
			else:
				map[c] +=1
		count = 0
		print(map)
		for key in map:
			if map[key] % 2!=0:
				count+=1
			if count >= 2:
				return False
		return True

	def missingNumber(self, nums):
		# 268
		# three answer first answer
		# res=len(nums)
		# for i in range(len(nums)):
		# 	print(f"res = {res}, i -num[i] = {i- nums[i]}, for {i} value {nums[i]}")
		# 	res+= (i -nums[i])
		# return res

		# #first answer	
		# b=sum(nums)
		# n = len(nums)
		# print((((n+1)*n)//2) - b )
		# n=(n*(n+1))//2
		# print(b, n)
		# return n-b

		# third answer mine
		nums.sort()
		nums +=[0]
		for i in range(0, len(nums)+1):
			print(i)
			if i != nums[i]:
				return i
			
	def closest_value(self, root, target):
		# 270 900
		print_tree_vertical(root)
		closest = root.val
		while root:
			closest = min(root.val, closest, key=lambda x : abs(target - x))
			root = root.left if root.val > target else root.right
		return closest
	
	def firstBadVersion(self, n):
		# 278
		def isBadVersion(x):
			return x >= 4
		l=1
		r=n
		while l<r:
			mid = (l+r)//2
			if isBadVersion(mid):
				r = mid
			else:
				l = mid+1
		return l




s = EasySolution()


# Problems till now 3600
test_arg1 = 5
test_arg2 = 0
passes = test_arg1
leetcode_output( 278,s.firstBadVersion, passes) #  // Output: 3
# print()



# leetcode_output(1,s.twoSum,[2,7,11,15], 9)   #  // output [0,1]
# leetcode_output(9,s.isPalindrome,123456) #   // output false
# leetcode_output(13,s.romanToInt,"III")  # // output 3
# leetcode_output(14,s.longestCommonPrefix,["aaa","aa","aaa"])  # // output aa
# leetcode_output(20,s.isValid,"(")  # // output True or False
# leetcode_output(21,s.mergeTwoLists,buildNodes([1, 2, 4]), buildNodes([1, 3, 4]))   #// output 1-> 1-> 2-> 3-> 4-> 4-> None
# leetcode_output(26,s.removeDuplicates,[1,1,2]) # // output [1,2, _, _, ...]
# leetcode_output(27,s.removeElement,[0,1,2,2,3,0,4,2], 2)  #// output 5, [0,1,3,0,4, _, _,_]
# leetcode_output(28,s.strStr,'sadbutsad','sad')  # // output 0
# leetcode_output(35,s.searchInsert,[1,3,5,6],7)  #// output 4
# leetcode_output(58,s.lengthOfLastWord,"   fly  me to   the   moon")  #// output 4
# leetcode_output(66,s.plusOne,[8,9,9])  #// output [9, 0, 0]
# leetcode_output(67,s.addBinary,'11', '1')  #// output 4 => 100
# leetcode_output(69,s.mySqrt,8), print( "Real One = ",int(8 ** .5))  #// output 2
# leetcode_output(70,s.climbStairs, 4)  #// output 5 =>fibinacci
# leetcode_output(83,s.deleteDuplicates, buildNodes([1,1,2,3,3])) # // output = 1 ->2 ->3 ->|

# leetcode_output(88,s.merge,[1,2,3,0,0,0], 3, [2,5,6], 3) # //output [1,2,2,3,5,6]
# leetcode_output(94,s.inorderTraversal,create_bst_tree(94)) #  // output [1,3,2] 
# leetcode_output(100,s.isSameTree,create_bst_tree(100), create_bst_tree(100)) #  // output true 
# leetcode_output(101,s.isSymmetric, create_bst_tree(101))#  // output true 
# leetcode_output(104,s.maxDepth,create_bst_tree(104)) #  // output 3
# leetcode_output(118,s.sortedArrayToBST, [-10, -3, 0,5,9]) #  // output Tree[[-10, -3, 0,5,9]]
# leetcode_output(110,s.isBalanced, create_bst_tree(110)) #  // output True
# leetcode_output(111,s.minDepth, create_bst_tree(111)) #  // output True
# leetcode_output(112,s.hasPathSum,create_bst_tree(112), 22) #  // output True
# leetcode_output(118,s.generate, 5) #  // output [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
# leetcode_output(119,s.getRow, 3) #  // output [1,3,3,1]
# leetcode_output(121,s.maxProfit, [7,1,5,3,6,4]) #  // output  5
# leetcode_output(125,s.validPalindrome, 'A man, a plan, a canal: Panama') #  // output  True
# leetcode_output(136,s.singleNumber, [2,3,4,1,4,3,2]) #  // output  1

 # hasCycle data
# tempNode2 = ListNode(2)
# tempNode4 = ListNode(-4)
# tempNode3 =ListNode(0,next=tempNode4)
# tempNode4.next = tempNode2
# tempNode2.next = tempNode3
# leetcode_output(141,s.hasCycle, ListNode(3,next=tempNode2)) #  // output  True

# leetcode_output(144,s.preorderTraversal, create_bst_tree(144)) #  // output  [1, 2, 4, 5, 6, 7, 3, 8, 9]
# leetcode_output(145,s.postorderTraversal, create_bst_tree(145)) #  // output   [4, 6, 7, 5, 2, 9, 8, 3, 1]


# read_buf = [] # read()   strign or file in helper file
# leetcode_output(157,s.read, read_buf, 8) #  // output ['l', 'e', 'e', 't', 'c', 'o', 'd', 'e']
# print(read_buf)


# point = buildNodes([8,4,5])
# leetcode_output( 160,s.getIntersectionNode,ListNode(4,next=ListNode(9,point)), ListNode(5,next=ListNode(6, next=ListNode(1, next=point)))) #  // output 8

# leetcode_output( 163,s.findMissingRanges,[0, 1,3 ,50, 75] , 0, 99) #  // output [[2,2], [4,49], [51,74], [76,99]]
# leetcode_output( 168,s.convertToTitle, 28) #  // output "AB"
# leetcode_output( 169,s.majorityElement,[8,8,7,7,7] ) #  // output 7
# leetcode_output( 170,s.twoSum3, [1,3,5,4,7], 8) #  // output True
# leetcode_output( 171,s.titleToNumber, "AAA") #  // output 703
# leetcode_output( 190,s.reverseBits, 43261596) #  // output 964176192 
# leetcode_output( 191,s.hammingWeight, 11) #  // output 3 
# leetcode_output( 202,s.isHappy, 19) #  // output True
# leetcode_output(203, s.removeElements, buildNodes([1,2,6,3,4,5,6]), 6) # // output = [1,2,3,4,5]
# leetcode_output( 205,s.isIsomorphic, "badc", 'baba') #  // output False
# leetcode_output( 217,s.reverseList, buildNodes([1,2,3,4,5])) #  // output [5,4,3,2,1] 
# leetcode_output( 217,s.containsDuplicate, [1,2,3,1]) #  // output True
# leetcode_output( 219,s.containsNearbyDuplicate, [1,2,3,1,2,3], 2) #  // output False
# leetcode_output( 222,s.countNodes,create_bst_tree(222) ) #  // output 6
# leetcode_output( 225,s.stackUsingQues,) #  // output ''
# leetcode_output( 226,s.invertTree, create_bst_tree(226)) #  // output  [4,7,2,9,6,3,1]
# leetcode_output( 228,s.summaryRanges, [0,1,2,4,5,7]) #  // output ['0->2', '4->5', '7']
# leetcode_output( 231,s.isPowerOfTwo, 16) #  // output True
# leetcode_output( 232,s.queUsingStacks) #  // output Class
# leetcode_output( 234,s.isPalindromeLinked, [1,1,2,3,4,3,2,1,1]) #  // output True
# leetcode_output( 242,s.isAnagram, 'aa', 'bb') #  // output False
# leetcode_output( 243,s.shortestDistance, ["practice", "makes", "perfect", "coding", "makes"], 'makes', 'coding') #  // output 1
# leetcode_output( 246,s.isStrobogrammatic, '0960') #  // output True
# leetcode_output( 252,s.can_attend_meetings, [Intervals(0,30),  Intervals(5,10), Intervals(15, 20)]) #  // output False
# leetcode_output( 257,s.binaryTreePaths, create_bst_tree(257)) #  // Output: ["1->2->5","1->3"]
# leetcode_output( 258,s.addDigits, 38) #  // Output: 3+8 = 11 = 1+1 == 2
# leetcode_output( 263,s.isUgly,18) #  // Output: Fasle
# leetcode_output( 266,s.can_permute_palindrome,"aab") #  // Output: True
# leetcode_output( 268,s.missingNumber,[0,2,1]) #  // Output: 3
# leetcode_output( 270,s.closest_value,create_bst_tree(270), 3.714286) #  // Output: 3