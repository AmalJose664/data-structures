

import time


class ListNode(object):
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next
	def __str__(self):
		return f"LNode-{self.val}"

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Intervals(object):
	def __init__(self, start, end):
		self.start = start
		self.end = end

def leetcode_output(number, output, *args, **kwargs):
	print("------------------------BEGIN----------------------------\n")
	start = time.time()
	out = output(*args, **kwargs)
	end = time.time() - start 
	print()
	print(f"Result = ", out)
	print(f"\nTime Took = {str(end*1000)[:6]} ms")
	print("Problem No. ",number)
	print("\n--------------------------END-----------------------------\n\n")
	return out





def create_bst_tree(problem: int):
    def tree_270():
        return TreeNode(4,
                        left=TreeNode(2, left=TreeNode(1), right=TreeNode(3)),
                        right=TreeNode(5))

    def tree_257():
        return TreeNode(1,
                        left=TreeNode(2, right=TreeNode(5)),
                        right=TreeNode(3))

    def tree_226():
        return TreeNode(4,
                        left=TreeNode(2, left=TreeNode(1), right=TreeNode(3)),
                        right=TreeNode(7, left=TreeNode(6), right=TreeNode(9)))

    def tree_222():
        return TreeNode(1,
                        left=TreeNode(2, left=TreeNode(4), right=TreeNode(5)),
                        right=TreeNode(3, left=TreeNode(6)))

    def tree_144_145():
        return TreeNode(1,
                        left=TreeNode(2,
                                      left=TreeNode(4),
                                      right=TreeNode(5, left=TreeNode(6), right=TreeNode(7))),
                        right=TreeNode(3, right=TreeNode(8, left=TreeNode(9))))

    def tree_112():
        return TreeNode(5,
                        left=TreeNode(4,
                                      left=TreeNode(11, left=TreeNode(7), right=TreeNode(2))),
                        right=TreeNode(8,
                                       left=TreeNode(13),
                                       right=TreeNode(4, right=TreeNode(1))))

    def tree_110_111_104():
        return TreeNode(3,
                        left=TreeNode(9),
                        right=TreeNode(20, left=TreeNode(15), right=TreeNode(7)))

    def tree_101():
        return TreeNode(1,
                        left=TreeNode(2, left=TreeNode(4), right=TreeNode(3)),
                        right=TreeNode(2, left=TreeNode(3), right=TreeNode(4)))

    def tree_94():
        return TreeNode(1, right=TreeNode(2, left=TreeNode(3)))

    def tree_100():
        return TreeNode(1, right=TreeNode(2, left=TreeNode(3)))

    tree_map = {
        270: tree_270,
        257: tree_257,
        226: tree_226,
        222: tree_222,
        144: tree_144_145,
        145: tree_144_145,
        112: tree_112,
        110: tree_110_111_104,
        111: tree_110_111_104,
        104: tree_110_111_104,
        101: tree_101,
        94:  tree_94,
        100: tree_100,
    }

    return tree_map.get(problem, lambda: None)()



def buildNodes(arr):
	dumy = ListNode()
	current = dumy
	for n in arr:
		current.next = ListNode(n)
		current = current.next
	return dumy.next

def showNodes(node):
	head = node
	while (head):
		print(head.val,end="-> ")
		head = head.next
	print("-|")
	return node


file_content = "leetcode"
file_pointer = 0
def read4(buf4: list) -> int:
    global file_content, file_pointer
    count = 0
    while count < 4 and file_pointer < len(file_content):
        buf4[count] = file_content[file_pointer]
        file_pointer += 1
        count += 1
    return count


def print_array_with_pointers(arr, l, r, endStr="", showLR=False):
	N=len(arr)
	
	if showLR:
		L="L"
		R="R"
	else:
		L=R="^"
	for num in arr:
		print(f"{num:>2}", end=" ")
	print(endStr)

	for i in range(N):
		if i == l and i == r:
			print(f"{L}{R}", end=" ")
		elif i == l:
			print(f" {L}", end=" ")
		elif i == r:
			print(f" {R}", end=" ")
		else:
			print("  ", end=" ")
	print("")
	print(f"|{'-'*((N*3)-2)}|")


def print_matrix(matrix):
	N = len(matrix)
	for i in range(N):
		for j in range(N):
			print(matrix[i][j], end=" ")
		print()

def print_tree_vertical(root):
	def _display_aux(root):
		
		if root.right is None and root.left is None:
			line = str(root.val)
			width = len(line)
			height = 1
			middle = width // 2
			return [line], width, height, middle

		# Only left child.
		if root.right is None:
			lines, n, p, x = _display_aux(root.left)
			s = str(root.val)
			u = len(s)
			first_line = (x + 1) * " " + (n - x - 1) * "_" + s
			second_line = x * " " + "/" + (n - x - 1 + u) * " "
			shifted_lines = [line + u * " " for line in lines]
			return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

		# Only right child.
		if root.left is None:
			lines, n, p, x = _display_aux(root.right)
			s = str(root.val)
			u = len(s)
			first_line = s + x * "_" + (n - x) * " "
			second_line = (u + x) * " " + "\\" + (n - x - 1) * " "
			shifted_lines = [u * " " + line for line in lines]
			return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

		# Two children.
		left, n, p, x = _display_aux(root.left)
		right, m, q, y = _display_aux(root.right)
		s = str(root.val)
		u = len(s)
		first_line = (x + 1) * " " + (n - x - 1) * "_" + s + y * "_" + (m - y) * " "
		second_line = x * " " + "/" + (n - x - 1 + u + y) * " " + "\\" + (m - y - 1) * " "
		if p < q:
			left += [" " * n] * (q - p)
		elif q < p:
			right += [" " * m] * (p - q)
		zipped_lines = zip(left, right)
		lines = [first + u * " " + second for first, second in zipped_lines]
		return [first_line, second_line] + lines, n + m + u, max(p, q) + 2, n + u // 2
	lines, *_ = _display_aux(root)
	for line in lines:
		print(line)
	
	return root