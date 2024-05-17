/* tar files - modified from https://github.com/ankitrohatgi/tarballjs/blob/master/tarball.js */
TarWriter = class {
	constructor() {
		this.fileData = [];
	}

	addFile(name, file, opts) {
		this.fileData.push({
			name: name,
			file: file,
			size: file.size,
			type: "file",
			dataType: "file",
			opts: opts
		});
	}

	_createBuffer() {
		let tarDataSize = 0;
		for(let i = 0; i < this.fileData.length; i++) {
			let size = this.fileData[i].size;
			tarDataSize += 512 + 512*Math.trunc(size/512);
			if(size % 512) {
				tarDataSize += 512;
			}
		}
		let bufSize = 10240*Math.trunc(tarDataSize/10240);
		if(tarDataSize % 10240) {
			bufSize += 10240;
		}
		this.buffer = new ArrayBuffer(bufSize);
	}

	write(onUpdate) {
		return new Promise((resolve,reject) => {
			this._createBuffer();
			let offset = 0;
			let filesAdded = 0;
			let onFileDataAdded = () => {
				filesAdded++;
				if (onUpdate) {
					onUpdate(filesAdded / this.fileData.length * 100);
				}
				if(filesAdded === this.fileData.length) {
					let arr = new Uint8Array(this.buffer);
					resolve(arr);
				}
			};
			for(let fileIdx = 0; fileIdx < this.fileData.length; fileIdx++) {
				let fdata = this.fileData[fileIdx];
				// write header
				this._writeFileName(fdata.name, offset);
				this._writeFileType(fdata.type, offset);
				this._writeFileSize(fdata.size, offset);
				this._fillHeader(offset, fdata.opts, fdata.type);
				this._writeChecksum(offset);

				// write file data
				let destArray = new Uint8Array(this.buffer, offset+512, fdata.size);
				if(fdata.dataType === "array") {
					for(let byteIdx = 0; byteIdx < fdata.size; byteIdx++) {
						destArray[byteIdx] = fdata.array[byteIdx];
					}
					onFileDataAdded();
				} else if(fdata.dataType === "file") {
					let reader = new FileReader();

					reader.onload = (function(outArray) {
						let dArray = outArray;
						return function(event) {
							let sbuf = event.target.result;
							let sarr = new Uint8Array(sbuf);
							for(let bIdx = 0; bIdx < sarr.length; bIdx++) {
								dArray[bIdx] = sarr[bIdx];
							}
							onFileDataAdded();
						};
					})(destArray);
					reader.readAsArrayBuffer(fdata.file);
				} else if(fdata.type === "directory") {
					onFileDataAdded();
				}

				offset += (512 + 512*Math.trunc(fdata.size/512));
				if(fdata.size % 512) {
					offset += 512;
				}
			}
		});
	}

	_writeString(str, offset, size) {
		let strView = new Uint8Array(this.buffer, offset, size);
		let te = new TextEncoder();
		if (te.encodeInto) {
			// let the browser write directly into the buffer
			let written = te.encodeInto(str, strView).written;
			for (let i = written; i < size; i++) {
				strView[i] = 0;
			}
		} else {
			// browser can't write directly into the buffer, do it manually
			let arr = te.encode(str);
			for (let i = 0; i < size; i++) {
				strView[i] = i < arr.length ? arr[i] : 0;
			}
		}
	}

	_writeFileName(name, header_offset) {
		// offset: 0
		this._writeString(name, header_offset, 100);
	}

	_writeFileType(typeStr, header_offset) {
		// offset: 156
		let typeChar = "0";
		if(typeStr === "file") {
			typeChar = "0";
		} else if(typeStr === "directory") {
			typeChar = "5";
		}
		let typeView = new Uint8Array(this.buffer, header_offset + 156, 1);
		typeView[0] = typeChar.charCodeAt(0);
	}

	_writeFileSize(size, header_offset) {
		// offset: 124
		let sz = size.toString(8);
		sz = this._leftPad(sz, 11);
		this._writeString(sz, header_offset+124, 12);
	}

	_leftPad(number, targetLength) {
		let output = number + '';
		while (output.length < targetLength) {
			output = '0' + output;
		}
		return output;
	}

	_writeFileMode(mode, header_offset) {
		// offset: 100
		this._writeString(this._leftPad(mode,7), header_offset+100, 8);
	}

	_writeFileUid(uid, header_offset) {
		// offset: 108
		this._writeString(this._leftPad(uid,7), header_offset+108, 8);
	}

	_writeFileGid(gid, header_offset) {
		// offset: 116
		this._writeString(this._leftPad(gid,7), header_offset+116, 8);
	}

	_writeFileMtime(mtime, header_offset) {
		// offset: 136
		this._writeString(this._leftPad(mtime,11), header_offset+136, 12);
	}

	_writeFileUser(user, header_offset) {
		// offset: 265
		this._writeString(user, header_offset+265, 32);
	}

	_writeFileGroup(group, header_offset) {
		// offset: 297
		this._writeString(group, header_offset+297, 32);
	}

	_writeChecksum(header_offset) {
		// offset: 148
		this._writeString("        ", header_offset+148, 8); // first fill with spaces

		// add up header bytes
		let header = new Uint8Array(this.buffer, header_offset, 512);
		let chksum = 0;
		for(let i = 0; i < 512; i++) {
			chksum += header[i];
		}
		this._writeString(chksum.toString(8), header_offset+148, 8);
	}

	_getOpt(opts, opname, defaultVal) {
		if(opts != null) {
			if(opts[opname] != null) {
				return opts[opname];
			}
		}
		return defaultVal;
	}

	_fillHeader(header_offset, opts, fileType) {
		let uid = this._getOpt(opts, "uid", 1000);
		let gid = this._getOpt(opts, "gid", 1000);
		let mode = this._getOpt(opts, "mode", fileType === "file" ? "664" : "775");
		let mtime = this._getOpt(opts, "mtime", Date.now());
		let user = this._getOpt(opts, "user", "tarballjs");
		let group = this._getOpt(opts, "group", "tarballjs");

		this._writeFileMode(mode, header_offset);
		this._writeFileUid(uid.toString(8), header_offset);
		this._writeFileGid(gid.toString(8), header_offset);
		this._writeFileMtime(Math.trunc(mtime/1000).toString(8), header_offset);

		this._writeString("ustar", header_offset+257,6); // magic string
		this._writeString("00", header_offset+263,2); // magic version

		this._writeFileUser(user, header_offset);
		this._writeFileGroup(group, header_offset);
	}
};

/* validate image sizes */
function validateJob(job, callbackOk, callbackError) {
	const reader = new FileReader()

	if(job.info.file.size > 20 * 1024 * 1024) {
		callbackError(job, 'Maxiální velikost souboru je 20MB.')
		return
	}

	reader.addEventListener('load', (event) => {
		const image = new Image();

		image.addEventListener('load', (event) => {
			const width = event.target.width
			const height = event.target.height

			if(width < 100 || height < 100) {
				callbackError(job, 'Minimální velikost obrázku je 100x100.')
			} else if(width > 10000 || height > 10000) {
				callbackError(job, 'Maximální velikost obrázku je 10000x10000.')
			} else {
				callbackOk(job)
			}
		})
		image.addEventListener('error', () => callbackError(job, 'Obrázek musí být ve formátu .png nebo .jpg'))
		image.src = event.target.result;
	})
	reader.addEventListener('error', () => callbackError(job, 'Nemůžu přečíst soubor.'))
	reader.readAsDataURL(job.info.image)
}

/* parse status check */
function parseCheckResult(checkResult) {
	if('error' in checkResult) {
		return {
			message: 'chyba při zpracování',
			details: checkResult.error,
			finished: true,
			download: false,
			error: true
		}
	} else {
		if(checkResult.status == 'queued') {
			return {
				message: `ve frontě na zpracování, pořadí ${checkResult.position}`,
				details: checkResult.info,
				jobId: checkResult.jobId,
				finished: false,
				download: false,
				error: false
			}
		} else if(checkResult.status == 'started') {
			if(checkResult.progress === undefined) {
				message ='zpracovávám, chvilku to potrvá'
			} else {
				message =`zpracovávám, ${Math.round(checkResult.progress * 100)}%`
			}

			return {
				message: message,
				details: checkResult.info,
				finished: false,
				download: false,
				error: false
			}
		} else if(checkResult.status == 'finished') {
			return {
				message: 'hotovo',
				details: checkResult.info,
				finished: true,
				download: checkResult.download === true,
				error: false
			}
		} else if(checkResult.status == 'canceled') {
			return {
				message: 'zrušeno',
				details: checkResult.info,
				finished: true,
				download: false,
				error: true
			}
		} else {
			return {
				message: 'chyba aplikace',
				details: checkResult.error,
				finished: true,
				download: false,
				error: true
			}
		}
	}
}

/* create info box with message */
function detailsBox(job, message, cls) {
	job.uiDetails.removeAttribute('class')
	job.uiDetails.classList.add('details')
	if(cls !== undefined) {
		job.uiDetails.classList.add(cls)
	}
	job.uiDetails.setAttribute('title', message)

	job.uiStatus.append(document.createTextNode('\u00A0'))
	job.uiStatus.append(job.uiDetails)
}

/* continue status checking */
function checkContinue(jobs, timeout) {
	if(Object.keys(jobs.jobIds).length > 0) {
		jobs.checkTimer = setTimeout(() => checkStatus(jobs), timeout)
	} else {
		jobs.checkTimer = null
	}
}

/* check error handler */
function checkError(jobs, message, details) {
	// iterate through job ids
	for(const [jobId, job] of Object.entries(jobs.jobIds)) {
		job.uiStatus.textContent = message
		if(details !== undefined) {
			detailsBox(job, details, 'error')
		}
	}

	checkContinue(jobs, 5000)
}

/* check processing status */
function checkStatus(jobs) {
	const connectionMessageError = 'chyba spojení, může to být dočasné'
	const xhttp = new XMLHttpRequest()

	// create and send check status request
	xhttp.addEventListener('load', (event) => {
		let parsedResponse

		// try to parse response
		try {
			parsedResponse = JSON.parse(event.target.response)
			if('error' in parsedResponse) {
				checkError(jobs, connectionMessageError, parsedResponse.error)
				return
			}
		} catch(error) {
			checkError(jobs, connectionMessageError)
			return
		}

		// iterate through job ids
		for(const [jobId, checkResult] of Object.entries(parsedResponse)) {
			const parsed = parseCheckResult(checkResult)
			const job = jobs.jobIds[jobId]

			job.uiStatus.textContent = parsed.message

			// add details ui
			if(parsed.details !== undefined) {
					detailsBox(job, parsed.details, parsed.error ? 'error' : 'info')
			}

			if(parsed.finished) {
				// finished, add download ui
				if(parsed.download) {
					job.uiStatus.append(document.createTextNode('\u00A0'))
					job.uiStatus.append(job.uiDownload)
					job.uiDownload.addEventListener('click', (event) => {
						const link = document.createElement('a')

						event.preventDefault()

						// virtual download link
						link.setAttribute('href', `download?id=${encodeURIComponent(job.jobId)}`)
						link.style.display = 'none'
						document.body.append(link)
						link.click()
						document.body.removeChild(link)

						// downloaded tick
						if(job.uiTick === undefined) {
							job.uiTick = document.createElement('span')
							job.uiTick.classList.add('tick')
							job.uiTick.innerHTML = imageTick
							job.uiTick.title = 'staženo';
							job.uiStatus.append(job.uiTick)
						}
					})
				}

				delete jobs.jobIds[jobId]
			}
		}

		// check if processing is finished
		checkContinue(jobs, 2000)
	}, false)

	// connection errors
	xhttp.addEventListener('error', event => checkError(jobs, connectionMessageError))
	xhttp.addEventListener('abort', event => checkError(jobs, connectionMessageError))
	xhttp.addEventListener('timeout', event => checkError(jobs, connectionMessageError))

	// send request
	xhttp.open('POST', 'status')
	xhttp.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded')
	xhttp.timeout = 60000
	xhttp.send(`ids=${encodeURIComponent(Object.keys(jobs.jobIds).join(','))}`)
}

/* upload complete callback */
function uploadComplete(event, job, jobs) {
	let parsedResponse

	// try to parse response
	try {
		parsedResponse = JSON.parse(event.target.response)
	} catch {
		job.uiStatus.textContent = 'chyba při komunikaci se serverem'
		return
	}

	parsed = parseCheckResult(parsedResponse)
	job.uiStatus.textContent = parsed.message
	if(parsed.details !== undefined) {
		detailsBox(job, parsed.details, parsed.error ? 'error' : undefined)
	}

	if(!parsed.error) {
		job.jobId = parsed.jobId
		jobs.jobIds[job.jobId] = job
		delete job.info.file
		delete job.info.image

		if(jobs.checkTimer === null) {
			checkContinue(jobs, 1000)
		}
	}
}

/* upload error callback */
function uploadError(event, job, jobs) {
	job.uiStatus.textContent = 'nepodařilo se soubor nahrát'
}

/* upload abort callback */
function uploadAbort(event, job, jobs) {
	job.uiStatus.textContent = 'nahrávání zrušeno'
}

/* upload progress callback */
function uploadProgress(event, job) {
	const percent = Math.round((event.loaded / event.total) * 100);

	job.uiStatus.textContent = `nahrávám, ${percent}%`
}

/* pick file from queue and process it */
function uploadQueue(jobs) {
	if(jobs.queue.length == 0) {
		uploadContinue(jobs)
		return
	}

	// get queue job and validate it
	validateJob(
		jobs.queue.shift(),
		(job) => {
			// send job
			const xhttp = new XMLHttpRequest()
			const formData = new FormData()

			formData.append('upload', job.info.file)
			xhttp.addEventListener('load', (event) => {
				uploadComplete(event, job, jobs)
				uploadContinue(jobs)
			}, false)
			xhttp.addEventListener('error', (event) => {
				uploadError(event, job, jobs)
				uploadContinue(jobs)
			}, false)
			xhttp.addEventListener('abort', (event) => {
				uploadAbort(event, job, jobs)
				uploadContinue(jobs)
			}, false)
			xhttp.upload.addEventListener('progress', (event) => {
				uploadProgress(event, job)
			}, false)
			xhttp.open('POST', 'upload')
			xhttp.send(formData)
		},
		(job, error) => {
			// show error
			job.uiStatus.textContent = 'chyba při zpracování'
			detailsBox(job, error, 'error')
			uploadContinue(jobs)
		}
	)
}

/* continue processing uploads */
function uploadContinue(jobs) {
	if(jobs.queue.length === 0) {
		jobs.queueTimer = null
	} else {
		jobs.queueTimer = setTimeout(() => uploadQueue(jobs), 1)
	}
}

/* submit form with chosen files */
async function submit(jobs) {
	const uiMessage = document.querySelector('#message')
	const uiWorkspace = document.querySelector('#workspace')
	let files = document.querySelector('#upload').files
	const maxFiles = 20

	uiMessage.textContent = ''


	if(files.length > maxFiles ) {
		uiMessage.textContent = `Maximální počet souborů, které můžeš nahrát najednou je ${maxFiles}.`
	} else {
		// create file groups to pair png with pgw and jpg with jgw files
		fileGroups = {}
		for(const file of files) {
			// split file name
			fileParts = file.name.split('.')
			if(fileParts.length == 1) {
				fileExtension = ''
				fileBaseName = file.name
			} else {
				fileExtension = fileParts[fileParts.length - 1]
				fileBaseName = fileParts.slice(0, fileParts.length - 1).join('.')
			}

			// add to groups
			const group = fileGroups[fileBaseName] || {}

			group[fileExtension] = file
			fileGroups[fileBaseName] = group
		}

		// recreate file list
		files = []
		for (const [fileBaseName, group] of Object.entries(fileGroups)) {
			if('pgw' in group) {
				if(!('png' in group)) {
					uiMessage.textContent = `Chybí nahrát png obrázek k ${group.pgw.name}.`
					return
				} else if(Object.keys(group).length > 2) {
					uiMessage.textContent = `Nelze jednoznačně přiřadit soubory k ${group.pgw.name}, očekává se pouze 1 png soubor stejného jména.`
					return
				}

				// tar png pair
				const tar = new TarWriter()

				tar.addFile(group.pgw.name, group.pgw)
				tar.addFile(group.png.name, group.png)
				const tarFile = await tar.write()
				files.push({
					title: `${group.png.name} + pgw`,
					file: new File([tarFile], `${group.png.name}.tar`, {type: 'application/x-tar'}),
					image: group.png
				})
			} else if('jgw' in group) {
				if(!('jpg' in group)) {
					uiMessage.textContent = `Chybí nahrát jpg obrázek k ${group.jgw.name}.`
					return
				} else if(Object.keys(group).length > 2) {
					uiMessage.textContent = `Nelze jednoznačně přiřadit soubory k ${group.jgw.name}, očekává se pouze 1 jpg soubor stejného jména.`
					return
				}

				// tar jpg pair
				const tar = new TarWriter()

				tar.addFile(group.jgw.name, group.jgw)
				tar.addFile(group.jpg.name, group.jpg)
				const tarFile = await tar.write()
				files.push({
					title: `${group.jpg.name} + jgw`,
					file: new File([tarFile], `${group.jpg.name}.tar`, {type: 'application/x-tar'}),
					image: group.jpg
				})
			} else {
				for (const [extension, file] of Object.entries(group)) {
					files.push({
						title: file.name,
						file: file,
						image: file
					})
				}
			}
		}

		document.querySelector('#workspace').style.display = 'table'

		// build result ui
		for(const info of files) {
			const uiRow = document.createElement('tr')
			const uiName = document.createElement('th')
			const uiStatus = document.createElement('td')
			const uiDetails = document.createElement('span')
			const uiDownload = document.createElement('a')
			const job = {}

			uiName.textContent = info.title
			uiStatus.textContent = 'čekám na nahrávání'
			uiDetails.textContent = 'info'
			uiDetails.setAttribute('tabindex', '0')
			uiDownload.classList.add('download')
			uiDownload.textContent = 'stáhnout'
			uiDownload.setAttribute('href', '#')
			uiRow.append(uiName)
			uiRow.append(uiStatus)
			uiWorkspace.append(uiRow)

			job.uiStatus = uiStatus
			job.uiDetails = uiDetails
			job.uiDownload = uiDownload
			job.info = info
			jobs.queue.push(job)
		}

		// start uploading
		if(jobs.queueTimer === null) {
			uploadContinue(jobs)
		}
	}
}

/* attach ui listeners */
function init(jobs) {
	document.querySelector('form').style.display = 'block'

	document.querySelector('input[type=submit]').style.display = 'none'
	document.querySelector('#upload').addEventListener('change', event => {
		submit(jobs).then()
	})
	document.querySelector('form').addEventListener('submit', event => {
		event.preventDefault()
		submit(jobs).then()
	})
}

// create graphics
const imageTick = '<svg fill="none" stroke="#4CAF50" version="1.1" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><g transform="matrix(1.9 0 0 2.3 -11 -15)"><path d="m17 7.4c0.31 0.28 0.34 0.75 0.06 1.1l-7.1 8c-0.14 0.16-0.35 0.25-0.56 0.25s-0.42-0.091-0.56-0.25l-2.9-3.2c-0.28-0.31-0.25-0.78 0.06-1.1 0.31-0.28 0.78-0.25 1.1 0.06l2.3 2.6 6.6-7.4c0.28-0.31 0.75-0.34 1.1-0.06z" clip-rule="evenodd" fill="#4caf50" fill-rule="evenodd"/></g></svg>'

// holds application state
const jobs = {
	queue: [],
	jobIds: [],
	checkTimer: null,
	queueTimer: null
}
