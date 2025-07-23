class HashTable {
	constructor(size = 5) {
		this.keyMap = new Array(size)
	}
	_hashFunction(key) {
		let sum = 0
		const PRIME_NUMBER = 31
		for (let i = 0; i < Math.min(key.length, 100); i++) {
			const charCode = key.charCodeAt(i) - 96
			sum = (sum * PRIME_NUMBER + charCode) % this.keyMap.length
		}
		return sum
	}
	set(key, value) {
		const index = this._hashFunction(key)

		if (!this.keyMap[index]) this.keyMap[index] = []
		this.keyMap[index].push([key, value])
		return this
	}
	get(key) {
		const index = this._hashFunction(key)
		if (this.keyMap[index]) {
			for (let i = 0; i < this.keyMap[index].length; i++) {
				if (this.keyMap[index][i][0] === key) {
					return this.keyMap[index][i][1]
				}
			}
		}
		return undefined
	}
	getAllKeys() {
		const keys = []

		for (let i = 0; i < this.keyMap.length; i++) {
			if (this.keyMap[i]) {
				for (let j = 0; j < this.keyMap[i].length; j++) {
					keys.push(this.keyMap[i][j][0])
				}
			}
		}
		return keys
	}
	getAllValues() {
		const values = []

		for (let i = 0; i < this.keyMap.length; i++) {
			if (this.keyMap[i]) {
				for (let j = 0; j < this.keyMap[i].length; j++) {
					values.push(this.keyMap[i][j][1])
				}
			}
		}
		return values
	}
}

const wordCounter = (text) => {
	const lowerText = text.toLowerCase()
	const wordMap = {}
	const words = lowerText.split(/\s+/)
	for (const word of words) {
		if (word in wordMap) {
			wordMap[word]++
		} else {
			wordMap[word] = 1
		}
	}

	return wordMap
}

// console.log(
//     wordCounter(
//         "success requires focus. focus brings clarity. fuels determination. without focus, goals fade. with follows. discipline strengthens daily leads to progress. on growth. learning. improvement. stay focused, driven"
//     )
// )

const twoSums = (nums = [], target) => {
	const numMap = {}
	for (let i = 0; i < nums.length; i++) {
		const compliment = target - nums[i]
		console.log("Compliment  = ", compliment)

		if (compliment in numMap && numMap[compliment] !== i) {
			return [numMap[compliment], i]
		}
		numMap[nums[i]] = i
		console.log(numMap)
	}
	return []
}
console.log(twoSums([2, 7, 11, 15], 9))
