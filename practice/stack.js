class Node {
	constructor(value) {
		this.value = value
		this.next = null
	}
}
class Stack {
	constructor(value) {
		const nNode = new Node(value)
		this.first = nNode
		this.length = 1
	}
	push(value) {
		let nNode = new Node(value)
		if (this.length === 0) {
			this.first = nNode
			this.length++
			return ""
		}

		nNode.next = this.first
		this.first = nNode
		this.length++
	}
	pop() {
		if (this.length === 0) {
			return undefined
		}
		let temp = this.first

		this.first = this.first.next
		temp.next = null
		this.length--

		return temp
	}
	min() {
		if (this.length === 0) {
			return undefined
		}
		let current = this.first
		let temp = current.value
		while (current) {
			if (temp > current.value) {
				temp = current.value
			}
			current = current.next
		}
		return temp
	}
}

let tStack = new Stack(3)

tStack.push(56)
tStack.push(36)
tStack.push(16)
tStack.push(59)
console.log("Min = ", tStack.min())
// console.log(tStack)
const isValidParathesis = (str) => {
	const stack = []
	const bra = {
		"{": "}",
		"[": "]",
		"(": ")",
	}
	for (let char of str) {
		if (bra[char]) {
			stack.push(char)
		} else {
			const top = stack.pop()
			if (!top || bra[top] !== char) {
				return false
			}
		}
	}
	return stack.length === 0
}
console.log(isValidParathesis("("))
const reverseString = (str) => {
	let stack = []
	let word = ""
	let newString = ""
	for (let chr of str) {
		//word = chr + word
		stack.push(chr)
	}
	while (stack.length > 0) {
		newString += stack.pop()
	}
	return newString
}
// console.log(
//     reverseString(
//         "thguoht fo niart s'rohtua eht wollof ot sredaer rof reisae ti gnikam ,krow nettirw erutcurts pleh shpargaraP .hpargarap txen eht ot snoitisnart ro sezirammus taht ecnetnes gnidulcnoc a semitemos dna ,cipot eht no etarobale taht secnetnes gnitroppus ,aedi niam eht secudortni taht ecnetnes cipot a sedulcni yllacipyt tI .tniop cificeps a poleved ot dezinagro ,aedi lartnec eno tuoba secnetnes fo puorg a si hpargarap A"
//     )
// )
