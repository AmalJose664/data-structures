function countDown(number = 100) {
	if (number === 0) {
		console.log("Function stop")
		return
	}
	console.log(number)
	countDown(number - 1)
}
function factorial(number = 5) {
	if (number === 0) return 1
	return number * factorial(number - 1)
}
console.log(factorial(4))
// countDown(6)
