const search = (query)=>{
	console.log("searching for "+query)
}

function debounce(fn, delay){
	let timer
	return function(...args){
		clearTimeout(timer)
		timer=setTimeout(()=>{
			fn(...args)
		}, delay)
	}
}

// const searchWithDebounce = debounce(search, 1000)
// searchWithDebounce("Hard ")
// searchWithDebounce("Hard js")
// searchWithDebounce("Hard js ques")
// searchWithDebounce("Hard js questions")
// searchWithDebounce("Hard js questions part2")

function throttle(fn, delay){
	let lastCall = 0
	return function(...args){
		const now = Date.now()
		if(now - lastCall < delay){
			return
		}
		lastCall = now
		return fn(...args)
	}
}

function sendChatMessage(message){
	console.log("Sending message ", message)
}

const sendChatMessageSlowMode = throttle(sendChatMessage, 2000)

sendChatMessageSlowMode(" HI")
sendChatMessageSlowMode(" HI how are u")
sendChatMessageSlowMode(" these")
sendChatMessageSlowMode(" messages")
sendChatMessageSlowMode(" will be ")
sendChatMessageSlowMode(" igonred")


setTimeout(()=>sendChatMessageSlowMode("New message after 2 seconds delay , this message will display .."), 2000)


