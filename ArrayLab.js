//1 -------------------------------------------------------------------
let oddNumbers = [1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20];
console.log("OG array: ", oddNumbers)
const checkForEven = (arr) => {
    let allOdd = true;
    arr.forEach((element, index) => {
        if (element % 2 === 0) {
            allOdd = false;
            arr.splice(index, 1, element + 1) //if number is even plus 1 to make odd
        }
    });
    return allOdd;
}
console.log(checkForEven(oddNumbers));
console.log("New Array with all odd nums", oddNumbers)


//another way thats cooler and easier to read :)
let oddNumbers2 = [1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20];

console.log("Old Array:", oddNumbers2);
const makeOddNumbers = (arr) => arr.map(num => num % 2 !== 0 ? num : num + 1);

let nowAllOdd = makeOddNumbers(oddNumbers2)
console.log("New Array with all odd nums:", nowAllOdd);



//2 -------------------------------------------------------------------
const fahrenheitTemps = [32, 50, 68, 86, 104, 122, 140, 158, 176, 194];

const clacAvgTemp = (arr) => {
    let total = 0;
    let count = 0;
    let largestNum = 0;

    arr.forEach((element) => {
        count++;
        total += element;
        if (largestNum < element) {
            largestNum = element;
        }
    });
    let avgTemp = total / count;

    return "Average temp is: " + avgTemp + " Highest temp is: " + largestNum;
}

console.log(clacAvgTemp(fahrenheitTemps));

//3 -------------------------------------------------------------------

const fruits = ["Apple", "Banana", "Orange", "Grapes", "Mango"];

const findFavFruit = (Array) => {

    let foundFruit = false;

    Array.forEach(element => {

        if (element === "kiwi") {
            foundFruit = true;
        }
    });
    if (!foundFruit) {
        Array.push("Kiwi");
    }
}

findFavFruit(fruits)
console.log(fruits)

//4 -------------------------------------------------------------------
const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"];

const shortenNames = () => {
    months.forEach((element,index) => {
        months[index] = element.slice(0,3);
    });
}
shortenNames(months);
console.log(months);


//5 -------------------------------------------------------------------
const mixedData = [42, 'hello', true, 3.14, 'world', false, null, undefined, 'goodbye', 7];

const onlyStrings = mixedData.filter(item => (typeof item === "string"));
console.log(onlyStrings);
