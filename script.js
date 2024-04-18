const imageElement = document.getElementById('displayedImage');
const classNameElement = document.getElementById('className');
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
	{ className: "Tennessee Warbler", imagePath: "data_samples/CUB/Tennessee Warbler.jpg", textPath: "gpt_descriptions/gpt4_0613_api_CUB/Tennessee Warbler.txt" },
	{ className: "737-200", imagePath: "data_samples/FGVCAircraft/737-200.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/737-200.txt" },
	{ className: "747-200", imagePath: "data_samples/FGVCAircraft/747-200.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/747-200.txt" },
	{ className: "767-400", imagePath: "data_samples/FGVCAircraft/767-400.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/767-400.txt" },
	{ className: "A310", imagePath: "data_samples/FGVCAircraft/A310.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A310.txt" },
	{ className: "A330-300", imagePath: "data_samples/FGVCAircraft/A330-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A330-300.txt" },
	{ className: "ATR-42", imagePath: "data_samples/FGVCAircraft/ATR-42.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/ATR-42.txt" },
	{ className: "C-130", imagePath: "data_samples/FGVCAircraft/C-130.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/C-130.txt" },
	{ className: "CRJ-700", imagePath: "data_samples/FGVCAircraft/CRJ-700.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/CRJ-700.txt" },
	{ className: "737-300", imagePath: "data_samples/FGVCAircraft/737-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/737-300.txt" },
	{ className: "747-300", imagePath: "data_samples/FGVCAircraft/747-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/747-300.txt" },
	{ className: "777-200", imagePath: "data_samples/FGVCAircraft/777-200.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/777-200.txt" },
	{ className: "A319", imagePath: "data_samples/FGVCAircraft/A319.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A319.txt" },
	{ className: "A340-200", imagePath: "data_samples/FGVCAircraft/A340-200.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A340-200.txt" },
	{ className: "BAE-125", imagePath: "data_samples/FGVCAircraft/BAE-125.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/BAE-125.txt" },
	{ className: "C-47", imagePath: "data_samples/FGVCAircraft/C-47.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/C-47.txt" },
	{ className: "CRJ-900", imagePath: "data_samples/FGVCAircraft/CRJ-900.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/CRJ-900.txt" },
	{ className: "737-600", imagePath: "data_samples/FGVCAircraft/737-600.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/737-600.txt" },
	{ className: "757-200", imagePath: "data_samples/FGVCAircraft/757-200.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/757-200.txt" },
	{ className: "777-300", imagePath: "data_samples/FGVCAircraft/777-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/777-300.txt" },
	{ className: "A321", imagePath: "data_samples/FGVCAircraft/A321.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A321.txt" },
	{ className: "A340-300", imagePath: "data_samples/FGVCAircraft/A340-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A340-300.txt" },
	{ className: "BAE 146-300", imagePath: "data_samples/FGVCAircraft/BAE 146-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/BAE 146-300.txt" },
	{ className: "Cessna 172", imagePath: "data_samples/FGVCAircraft/Cessna 172.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/Cessna 172.txt" },
	{ className: "737-800", imagePath: "data_samples/FGVCAircraft/737-800.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/737-800.txt" },
	{ className: "767-300", imagePath: "data_samples/FGVCAircraft/767-300.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/767-300.txt" },
	{ className: "A300B4", imagePath: "data_samples/FGVCAircraft/A300B4.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A300B4.txt" },
	{ className: "A330-200", imagePath: "data_samples/FGVCAircraft/A330-200.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A330-200.txt" },
	{ className: "A340-500", imagePath: "data_samples/FGVCAircraft/A340-500.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/A340-500.txt" },
	{ className: "Beechcraft 1900", imagePath: "data_samples/FGVCAircraft/Beechcraft 1900.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/Beechcraft 1900.txt" },
	{ className: "Cessna 525", imagePath: "data_samples/FGVCAircraft/Cessna 525.jpg", textPath: "gpt_descriptions/gpt4_0613_api_FGVCAircraft/Cessna 525.txt" },
	{ className: "2000 AM General Hummer SUV", imagePath: "data_samples/StanfordCars/2000 AM General Hummer SUV.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2000 AM General Hummer SUV.txt" },
	{ className: "2012 Aston Martin V8 Vantage Coupe", imagePath: "data_samples/StanfordCars/2012 Aston Martin V8 Vantage Coupe.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Aston Martin V8 Vantage Coupe.txt" },
	{ className: "2007 BMW 6 Series Convertible", imagePath: "data_samples/StanfordCars/2007 BMW 6 Series Convertible.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2007 BMW 6 Series Convertible.txt" },
	{ className: "2012 Audi A5 Coupe", imagePath: "data_samples/StanfordCars/2012 Audi A5 Coupe.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Audi A5 Coupe.txt" },
	{ className: "2007 Cadillac Escalade EXT Crew Cab", imagePath: "data_samples/StanfordCars/2007 Cadillac Escalade EXT Crew Cab.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2007 Cadillac Escalade EXT Crew Cab.txt" },
	{ className: "2012 Audi S5 Convertible", imagePath: "data_samples/StanfordCars/2012 Audi S5 Convertible.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Audi S5 Convertible.txt" },
	{ className: "2007 Chevrolet Corvette Ron Fellows Edition Z06", imagePath: "data_samples/StanfordCars/2007 Chevrolet Corvette Ron Fellows Edition Z06.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2007 Chevrolet Corvette Ron Fellows Edition Z06.txt" },
	{ className: "2012 BMW 3 Series Sedan", imagePath: "data_samples/StanfordCars/2012 BMW 3 Series Sedan.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 BMW 3 Series Sedan.txt" },
	{ className: "2007 Chevrolet Impala Sedan", imagePath: "data_samples/StanfordCars/2007 Chevrolet Impala Sedan.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2007 Chevrolet Impala Sedan.txt" },
	{ className: "2012 BMW 3 Series Wagon", imagePath: "data_samples/StanfordCars/2012 BMW 3 Series Wagon.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 BMW 3 Series Wagon.txt" },
	{ className: "2007 Chevrolet Silverado 1500 Classic Extended Cab", imagePath: "data_samples/StanfordCars/2007 Chevrolet Silverado 1500 Classic Extended Cab.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2007 Chevrolet Silverado 1500 Classic Extended Cab.txt" },
	{ className: "2012 BMW X6 SUV", imagePath: "data_samples/StanfordCars/2012 BMW X6 SUV.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 BMW X6 SUV.txt" },
	{ className: "2007 Dodge Caliber Wagon", imagePath: "data_samples/StanfordCars/2007 Dodge Caliber Wagon.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2007 Dodge Caliber Wagon.txt" },
	{ className: "2012 BMW Z4 Convertible", imagePath: "data_samples/StanfordCars/2012 BMW Z4 Convertible.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 BMW Z4 Convertible.txt" },
	{ className: "2008 Audi RS 4 Convertible", imagePath: "data_samples/StanfordCars/2008 Audi RS 4 Convertible.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2008 Audi RS 4 Convertible.txt" },
	{ className: "2012 Buick Regal GS", imagePath: "data_samples/StanfordCars/2012 Buick Regal GS.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Buick Regal GS.txt" },
	{ className: "2009 Chevrolet TrailBlazer SS", imagePath: "data_samples/StanfordCars/2009 Chevrolet TrailBlazer SS.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2009 Chevrolet TrailBlazer SS.txt" },
	{ className: "2012 Cadillac CTS-V Sedan", imagePath: "data_samples/StanfordCars/2012 Cadillac CTS-V Sedan.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Cadillac CTS-V Sedan.txt" },
	{ className: "2009 Dodge Sprinter Cargo Van", imagePath: "data_samples/StanfordCars/2009 Dodge Sprinter Cargo Van.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2009 Dodge Sprinter Cargo Van.txt" },
	{ className: "2012 Chevrolet Camaro Convertible", imagePath: "data_samples/StanfordCars/2012 Chevrolet Camaro Convertible.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Chevrolet Camaro Convertible.txt" },
	{ className: "2010 Chevrolet Cobalt SS", imagePath: "data_samples/StanfordCars/2010 Chevrolet Cobalt SS.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2010 Chevrolet Cobalt SS.txt" },
	{ className: "2012 Chevrolet Corvette Convertible", imagePath: "data_samples/StanfordCars/2012 Chevrolet Corvette Convertible.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Chevrolet Corvette Convertible.txt" },
	{ className: "2010 Dodge Ram Pickup 3500 Crew Cab", imagePath: "data_samples/StanfordCars/2010 Dodge Ram Pickup 3500 Crew Cab.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2010 Dodge Ram Pickup 3500 Crew Cab.txt" },
	{ className: "2012 Chevrolet Corvette ZR1", imagePath: "data_samples/StanfordCars/2012 Chevrolet Corvette ZR1.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Chevrolet Corvette ZR1.txt" },
	{ className: "2011 Audi S6 Sedan", imagePath: "data_samples/StanfordCars/2011 Audi S6 Sedan.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2011 Audi S6 Sedan.txt" },
	{ className: "2012 Chevrolet Silverado 2500HD Regular Cab", imagePath: "data_samples/StanfordCars/2012 Chevrolet Silverado 2500HD Regular Cab.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Chevrolet Silverado 2500HD Regular Cab.txt" },
	{ className: "2011 Dodge Challenger SRT8", imagePath: "data_samples/StanfordCars/2011 Dodge Challenger SRT8.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2011 Dodge Challenger SRT8.txt" },
	{ className: "2012 Dodge Caliber Wagon", imagePath: "data_samples/StanfordCars/2012 Dodge Caliber Wagon.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Dodge Caliber Wagon.txt" },
	{ className: "2012 Acura TSX Sedan", imagePath: "data_samples/StanfordCars/2012 Acura TSX Sedan.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Acura TSX Sedan.txt" },
	{ className: "2012 Dodge Charger Sedan", imagePath: "data_samples/StanfordCars/2012 Dodge Charger Sedan.jpg", textPath: "gpt_descriptions/gpt4_0613_api_StanfordCars/2012 Dodge Charger Sedan.txt" }
];

function loadRandomData() {
    const randomIndex = Math.floor(Math.random() * data.length);
    const selectedItem = data[randomIndex];

    fetch(selectedItem.imagePath)
        .then(response => {
            imageElement.src = response.url;
            // Set the class name by removing the path and file extension
            const name = selectedItem.className; // Assuming the className attribute is the clean name
            classNameElement.textContent = name;
        });

    fetch(selectedItem.textPath)
        .then(response => response.text())
        .then(text => {
            const sentences = text.split(/\r?\n/);
            const randomSentence = sentences[Math.floor(Math.random() * sentences.length)];
            textElement.textContent = randomSentence;
        });
}

window.onload = loadRandomData; // Load random data on page load