"use strict";

const puppeteer = require("puppeteer");
const fs = require("fs");
const readline = require("readline");
const { strict } = require("assert");
const { stringify } = require("querystring");


const name_exp = process.argv[2];
const path_base = '../drnn/experiments'
const split = process.argv[3];
const out_dir_img = path_base + "/" + name_exp + "/test/" + split + "_img";


if (!fs.existsSync(out_dir_img)) {
    fs.mkdir(out_dir_img, (err) => {
        if (err) {
            throw err;
        }
        //console.log(out_dir_img);
    });
}

var name_idx = 0;
var browser;
//const MAX_HEIGHT = 1440 * 16;
//const MAX_HEIGHT = 16384;
const MAX_HEIGHT = 14000;

// helper
async function scrollToBottom(page, viewportHeight) {
    console.log("=== 42 scrollToBottom");
    const getScrollHeight = () => {
        return Promise.resolve(document.documentElement.scrollHeight);
    };
    console.log("=== 46 get height");

    let scrollHeight = await page.evaluate(getScrollHeight);
    let currentPosition = 0;
    let scrollNumber = 0;

    if (scrollHeight > MAX_HEIGHT) {
        return scrollHeight;
    }
    console.log("=== 55 scroll begin");

    while (currentPosition < scrollHeight) {
        scrollNumber += 1;
        const nextPosition = scrollNumber * viewportHeight;
        console.log("=== 60 await 1");
        await page.evaluate(function (scrollTo) {
            return Promise.resolve(window.scrollTo(0, scrollTo));
        }, nextPosition);
        console.log("=== 60 await 2");
        await page
            .waitForNavigation({ waitUntil: "networkidle2", timeout: 1500 })
            .catch((e) => console.log("timeout exceed. proceed to next operation"));
        console.log("=== 68 after wait");

        currentPosition = nextPosition;
        //console.log(`scrollNumber: ${scrollNumber}`);
        //console.log(`currentPosition: ${currentPosition}`);

        // 2
        /*
        scrollHeight = await page.evaluate(getScrollHeight);
        console.log(`ScrollHeight ${scrollHeight}`);
        if (scrollHeight > MAX_HEIGHT) {
          return scrollHeight;
        }
        */
        if (currentPosition > MAX_HEIGHT) {
            return currentPosition;
        }
    }
    console.log("=== 80 end scroll");
    return scrollHeight;
}


async function autoScroll(page) {
    //console.log("=== 92 autoScroll");


    return await page.evaluate(async () => {
        //console.log("=== 96 eval");


        return await new Promise((resolve, reject) => {

            //console.log("=== 98 promise");
            var totalHeight = 0;
            //var scrollHeight = 17000;
            var distance = 100;
            var timer = setInterval(() => {
                //console.log("=== 97 getheight");
                var scrollHeight = document.body.scrollHeight;
                //console.log("=== 99 gotheight");
                window.scrollBy(0, distance);
                //console.log("=== 101 scroll");
                totalHeight += distance;
                //console.log(totalHeight);

                if (totalHeight >= scrollHeight) {
                    clearInterval(timer);
                    //console.log("=== 111 resolve");
                    resolve(totalHeight);
                }

            }, 100);
        });

    });


}

//define crawler
async function run(url, idx) {


    await (async () => {
        try {
            const viewportHeight = 1440; //2880;
            const viewportWidth = 1440;
            //console.log("start")

            //console.log("puppeteer launch");
            browser = await puppeteer.launch({
                //headless: false,
                //slowMo: 300,
                //defaultViewport: {
                //  width: viewportWidth,
                //  height: viewportHeight,
                //},
            }).catch(e => {
                console.log("launch ", e)
            });

            const page = await browser.newPage().catch(e => {
                console.log("new page", e)
            });
            await page.setViewport({ width: viewportWidth, height: viewportHeight, deviceScaleFactor: 0 }).catch(e => {
                console.log("set viewport", e)
            });

            /*
            page.on('console', msg => {
              for (let i = 0; i < msg._args.length; ++i)
                console.log(`${i}: ${msg._args[i]}`);
            });
            */


            //console.log("=== 101: goto url");
            //await page.setDefaultNavigationTimeout(0);
            await page.goto(url, { waitUntil: "domcontentloaded", timeout: 5000 }).catch(e => {
                console.log("goto url", e)
            });
            //console.log("=== 104: goto end");
            //await page.goto(url);

            await page._client.send("DOM.enable").catch(e => {
                console.log("dom enable", e)
            });
            //console.log("=== 108 dom enable");
            await page._client.send("CSS.enable").catch(e => {
                console.log("css enable", e)
            });
            //console.log("=== 110 css enable");


            //scroll
            //TODO: スクショの下端がループして上端になる
            await page
                .waitForNavigation({ waitUntil: "networkidle2", timeout: 3000 })
                .catch((e) => console.log("timeout exceed. proceed to next operation"));
            //console.log("=== 117 wait navigation");

            //var h = await scrollToBottom(page, viewportHeight);
            var h = new Promise(async (resolve, reject) => {
                setTimeout(() => {
                    reject(new Error("timeout"))
                }, 10000);
                resolve(await autoScroll(page));
            })
            //var h = await autoScroll(page);

            //console.log("=== 120 after scroll 1");
            if (h > MAX_HEIGHT) {
                throw new Error("too large, skip");
            }
            //console.log("=== 123 after scroll 2");



            /*
            await page.evaluate((_) => {
              window.scrollBy({
                top: 800,
                behavior: "smooth",
              });
            });
      
            // Arbitrary wait to allow things to load
            await wait(1000);
      
            await page.evaluate((_) => {
              window.scrollBy({
                top: MAX_HEIGHT,
                behavior: "smooth",
              });
            });
      
            // Another arbitrary wait to allow more things to load
            await wait(1000);
      
            // Scroll back to top
            await page.evaluate((_) => {
              window.scrollTo({
                top: 0,
                behavior: "smooth",
              });
            });
      
            // A full height viewport reveal to force any missing elements to reveal.
            // We set here to 10000 but it can be set to the real calculated height or
            // Something even larger like 20000 or 30000
            //await page.setViewport({ width: 1440, height: MAX_HEIGHT });
            await page.setViewport({
              width: 1440,
              height: await page.evaluate(() => document.body.clientHeight)
            });
            */



            /*
            // check if a selector exists
            const target = await page.evaluate(() => {
              const tag_target = "form"
              var tgt = document.getElementsByTagName(tag_target)[0];
      
              return tgt;
            });
            //console.log(target);
            if (target === undefined) {
              throw new Error("no target, skip");
            } else {
              console.log("found! ", url);
            }
            */


            await page.screenshot({
                path: out_dir_img + "/" + String(idx) + ".jpg",
                fullPage: true,
                timeout: 200,
                type: "jpeg",
                /*
                clip: {
                  x: 0,
                  y: 0,
                  width: 1440,
                  height: MAX_HEIGHT,
                }
                */
                //clip: { x: 0, y: 0, viewportWidth, h }
            }).catch(e => {
                console.log("screenshot.close", e)
            });
            //console.log("screenshot taken");



        } catch (error) {

            console.log(error);
        } finally {
            if (browser) {
                //console.log("browser close");
                await browser.close().catch(e => {
                    console.log("browser.close", e)
                });
            }
        }
    })().catch(e => {
        console.log("all", e)
    });
}



var walk = require('walk');
var files = [];

var path_html = path_base + "/" + name_exp + "/test/" + split
var walker = walk.walk(path_html, { followLinks: false });


walker.on('file', function (root, stat, next) {
    // Add this file to the list of files
    files.push(root + '/' + stat.name);
    next();
});

walker.on('end', function () {
    console.log(files.length);

    var idx = 0;
    const path = require('path');
    console.log("start");
    (async () => {
        while (idx < files.length) {
            var name = path.basename(files[idx])
            const url = "http://0.0.0.0:8080/" + name;
            await run(url, name);
            idx += 1;
        }
    })();
    console.log("end");
});
