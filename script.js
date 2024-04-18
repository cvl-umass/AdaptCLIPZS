const imageElement = document.getElementById('displayedImage');
const textElement = document.getElementById('displayText');

const data = [
    { className: "Black footed Albatross", imagePath: "data_samples/CUB/Black footed Albatross.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Black footed Albatross.txt" },
	{ className: "Golden winged Warbler", imagePath: "data_samples/CUB/Golden winged Warbler.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Golden winged Warbler.txt" },
	{ className: "Nighthawk", imagePath: "data_samples/CUB/Nighthawk.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Nighthawk.txt" },
	{ className: "Western Gull", imagePath: "data_samples/CUB/Western Gull.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Western Gull.txt" },
	{ className: "Blue headed Vireo", imagePath: "data_samples/CUB/Blue headed Vireo.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Blue headed Vireo.txt" },
	{ className: "Gray Kingbird", imagePath: "data_samples/CUB/Gray Kingbird.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Gray Kingbird.txt" },
	{ className: "Red faced Cormorant", imagePath: "data_samples/CUB/Red faced Cormorant.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Red faced Cormorant.txt" },
	{ className: "Western Wood Pewee", imagePath: "data_samples/CUB/Western Wood Pewee.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Western Wood Pewee.txt" },
	{ className: "Chuck will Widow", imagePath: "data_samples/CUB/Chuck will Widow.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Chuck will Widow.txt" },
	{ className: "Great Grey Shrike", imagePath: "data_samples/CUB/Great Grey Shrike.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Great Grey Shrike.txt" },
	{ className: "Red winged Blackbird", imagePath: "data_samples/CUB/Red winged Blackbird.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Red winged Blackbird.txt" },
	{ className: "Whip poor Will", imagePath: "data_samples/CUB/Whip poor Will.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Whip poor Will.txt" },
	{ className: "Clay colored Sparrow", imagePath: "data_samples/CUB/Clay colored Sparrow.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Clay colored Sparrow.txt" },
	{ className: "Hooded Merganser", imagePath: "data_samples/CUB/Hooded Merganser.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Hooded Merganser.txt" },
	{ className: "Rufous Hummingbird", imagePath: "data_samples/CUB/Rufous Hummingbird.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Rufous Hummingbird.txt" },
	{ className: "White necked Raven", imagePath: "data_samples/CUB/White necked Raven.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/White necked Raven.txt" },
	{ className: "Crested Auklet", imagePath: "data_samples/CUB/Crested Auklet.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Crested Auklet.txt" },
	{ className: "Horned Grebe", imagePath: "data_samples/CUB/Horned Grebe.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Horned Grebe.txt" },
	{ className: "Scissor tailed Flycatcher", imagePath: "data_samples/CUB/Scissor tailed Flycatcher.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Scissor tailed Flycatcher.txt" },
	{ className: "Winter Wren", imagePath: "data_samples/CUB/Winter Wren.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Winter Wren.txt" },
	{ className: "Elegant Tern", imagePath: "data_samples/CUB/Elegant Tern.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Elegant Tern.txt" },
	{ className: "Horned Lark", imagePath: "data_samples/CUB/Horned Lark.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Horned Lark.txt" },
	{ className: "Sooty Albatross", imagePath: "data_samples/CUB/Sooty Albatross.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Sooty Albatross.txt" },
	{ className: "Worm eating Warbler", imagePath: "data_samples/CUB/Worm eating Warbler.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Worm eating Warbler.txt" },
	{ className: "European Goldfinch", imagePath: "data_samples/CUB/European Goldfinch.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/European Goldfinch.txt" },
	{ className: "Mangrove Cuckoo", imagePath: "data_samples/CUB/Mangrove Cuckoo.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Mangrove Cuckoo.txt" },
	{ className: "Swainson Warbler", imagePath: "data_samples/CUB/Swainson Warbler.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Swainson Warbler.txt" },
	{ className: "Fox Sparrow", imagePath: "data_samples/CUB/Fox Sparrow.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Fox Sparrow.txt" },
	{ className: "Mourning Warbler", imagePath: "data_samples/CUB/Mourning Warbler.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Mourning Warbler.txt" },
	{ className: "Tennessee Warbler", imagePath: "data_samples/CUB/Tennessee Warbler.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Tennessee Warbler.txt" }

    // Add all class names and paths accordingly
];

function loadRandomData() {
    const randomIndex = Math.floor(Math.random() * data.length);
    const selectedItem = data[randomIndex];

    fetch(selectedItem.imagePath)
        .then(response => imageElement.src = response.url);

    fetch(selectedItem.textPath)
        .then(response => response.text())
        .then(text => {
            const sentences = text.split(/\r?\n/);
            const randomSentence = sentences[Math.floor(Math.random() * sentences.length)];
            textElement.textContent = randomSentence;
        });
}

window.onload = loadRandomData; // Load random data on page load
