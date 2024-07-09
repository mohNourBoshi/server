// ==UserScript==
// @name         solver
// @namespace    http://tampermonkey.net/
// @version      2024-07-07
// @description  try to take over the world!
// @author       You
// @match        https://www.ecsc.gov.sy/requests/info/*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=gov.sy
// @grant        none
// ==/UserScript==

(function () {
    'use strict';

    function elementExists(selector) {
        return document.querySelector(selector) !== null;
    }

    function waitForElement(selector, callback) {
        const observer = new MutationObserver(() => {
            if (elementExists(selector)) {
                observer.disconnect();
                callback();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    waitForElement("app-nav-bar > mat-toolbar > div:nth-child(2) > ul li", () => {
        document.querySelector("app-nav-bar > mat-toolbar > div:nth-child(2) > ul li").innerHTML = `
                <input type="checkbox" id="vehicle1" name="vehicle1" value="1" checked>
                <label for="vehicle1"> solve when it show up</label>`;


        async function sendImage(base64) {
            try {
                let res = await fetch("http://127.0.0.1:5000/image", {
                    cache: "no-store",
                    headers: {
                        "accept": "*/*",
                        "accept-language": "en-US,en;q=0.9",
                        "content-type": "application/json"
                    },
                    referrerPolicy: "strict-origin-when-cross-origin",
                    body: JSON.stringify({ base64_image: `data:image/jpeg;base64,${base64}` }),
                    method: "POST",
                    mode: "cors",
                    credentials: "omit"
                });
                if (res.status === 200) {
                    let imageres = await res.json();
                    imageres = imageres.solvetasks.map(item => item.class_name.trim());
                    return formatSolve(imageres);
                } else {
                    console.error(`Unexpected status code: ${res.status}`);
                    return ['1', '+', '1'];
                }
            } catch (error) {
                console.error("Error occurred in sendImage:", error);
                return ['1', '+', '1'];
            }
        }

        function formatSolve(solve) {
            console.log(`Input solve: ${solve}`);
            if (solve.includes(null)) {
                console.log("Null values in solve array");

                return ['1', '+', '1'].join('');
            }

            let nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            console.log(`this is log from format this in put is ${solve}`)
            if (solve.includes(null)) {
                console.log("wow it is all null ");
            } else if (['*', '-', '+'].includes(solve[1])) {
                console.log("it is ok");
            }
            else if (['*', '-', '+'].includes(solve[2])) {
                solve = solve[1] + solve[2] + solve[0]
            }
            else if (['*', '-', '+'].includes(solve[0])) {
                solve = solve[2] + solve[0] + solve[1]
            }
            else if (nums.includes(solve[0]) & nums.includes(solve[1]) & nums.includes(solve[2])) {
                if (nums.includes(solve[0]) & ['1'].includes(solve[1]) & nums.includes(solve[2])) {
                    solve = solve[0] + '+' + solve[2]

                }
                else if (nums.includes(solve[0]) & nums.includes(solve[1]) & ['1'].includes(solve[2])) {
                    solve = solve[1] + '+' + solve[0]

                }
                else if (['1'].includes(solve[0]) & nums.includes(solve[1]) & nums.includes(solve[2])) {
                    solve = solve[2] + '+' + solve[1]

                } else {
                    solve[1] = '+'
                }
            }
            else if (solve[0] == null | solve[1] == null | solve[2] == null) {
                solve = ['1', '+', '1']
            }


            return solve.join('');
        }

        async function addMathEvaluator() {
            if (document.querySelector("#vehicle1")?.checked) {
                const imageSrc = document.querySelector("body > div.swal2-container.swal2-rtl.swal2-center.swal2-backdrop-show > div > img")?.src;
                if (imageSrc) {
                    console.time('solve image');
                    const solve = await sendImage(imageSrc);
                    console.timeEnd('solve image');
                    console.log(`Solved expression: ${solve}`);
                    const mathInput = document.querySelector("#mathInput");
                    if (mathInput) {
                        mathInput.value = solve;
                        document.querySelector("#mathForm button").click();
                    }
                }
            }
        }

        function observeAndRun(selector) {
            const observer = new MutationObserver((mutations, obs) => {
                if (elementExists(selector)) {
                    obs.disconnect();
                    addMathEvaluator();
                    observeAndRun(selector); // Re-run the observer
                }
            });

            observer.observe(document.body, { childList: true, subtree: true });
        }

        // Start observing for the target element
        observeAndRun("div.swal2-container.swal2-rtl.swal2-center.swal2-backdrop-show");
    });

})();
