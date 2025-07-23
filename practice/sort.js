function bubbleSort(unsortedArray = []) {
	for (let i = unsortedArray.length - 1; i > 0; i--) {
		for (let j = 0; j < i; j++) {
			if (unsortedArray[j] > unsortedArray[j + 1]) {
				let temp = unsortedArray[j]
				unsortedArray[j] = unsortedArray[j + 1]
				unsortedArray[j + 1] = temp
			}
		}
	}
	return unsortedArray
}
function selectionSort(arr = []) {
	for (let i = 0; i < arr.length - 1; i++) {
		let minIndex = i
		for (let j = i + 1; j < arr.length; j++) {
			if (arr[j] < arr[minIndex]) {
				minIndex = j
			}
		}
		if (i !== minIndex) {
			let temp = arr[i]
			arr[i] = arr[minIndex]
			arr[minIndex] = temp
		}
	}
	return arr
}
function insertionSort(arr = []) {
	for (i = 1; i < arr.length; i++) {
		let key = arr[i]
		let j = i - 1
		while (j >= 0 && arr[j] > key) {
			arr[j + 1] = arr[j]
			j--
		}
		arr[j + 1] = key
	}
	return arr
}

const merge = (left = [], right = []) => {
	const result = []
	let i = 0
	let j = 0
	while (i < left.length && j < right.length) {
		if (left[i] < right[j]) {
			result.push(left[i])
			i++
		} else {
			result.push(right[j])
			j++
		}
	}
	result.push(...left.slice(i))
	result.push(...right.slice(j))

	return result
}
function mergeSort(arr = []) {
	if (arr.length <= 1) return arr
	const middle = Math.floor(arr.length / 2)
	const left = arr.slice(0, middle)
	const right = arr.slice(middle)

	return merge(mergeSort(left), mergeSort(right))
}
const newSort = (arr = []) => {
	for (let i = 0; i < arr.length; i++) {
		setTimeout(() => {
			console.log(arr[i])
		}, arr[i])
	}
}

console.log(mergeSort([38, 27, 43, 3, 9, 82, 10]))
