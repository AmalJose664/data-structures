//--------------------------------------------------------------------------------------------------------
const groceris = ["milk", "bread", "eggs", "flour", "choose", "sugar"]

const searchForItem = (item) => {
    // o(n) time
    for (let i = 0; i < groceris.length; i++) {
        if (groceris[i] === item) {
            return console.log("Found item at " + i)
        }
    }
    return console.log("Not Found")
}
//--------------------------------------------------------------------------------------------------------
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
// o(1) time
const getElement = (arr, index) => arr[index]
//--------------------------------------------------------------------------------------------------------

function findPairs(arr) {
    // o(n^2)
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length; j++) {
            console.log(`Pair: ${arr[i]}, ${arr[j]}`)
        }
    }
    // o(n)
    for (let q = 0; q < arr.length; q++) {
        console.log("===============> ", q)
    }
    return "Finished" // final o(n^2) removed dominant tern n = o(n), so o(n^2)
}

//--------------------------------------------------------------------------------------------------------

// o(log n)
