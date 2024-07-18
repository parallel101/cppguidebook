const puppeteer = require('puppeteer');
var args = process.argv.slice(2);
var url = args[0];
var pdfPath = args[1];
var title = args[2];

console.log('Saving', url, 'to', pdfPath);

// date –  formatted print date
// title – document title
// url  – document location
// pageNumber – current page number
// totalPages – total pages in the document
headerHtml = `
<div style="font-size: 10px; padding-right: 1em; text-align: right; width: 100%;">
    <span>${title}</span>  <span class="pageNumber"></span> / <span class="totalPages"></span>
</div>`;

footerHtml = ` `;


(async() => {
    const browser = await puppeteer.launch({
        headless: true,
        executablePath: process.env.CHROME_BIN || null,
        args: ['--no-sandbox', '--headless', '--disable-gpu', '--disable-dev-shm-usage', '--lang=zh-CN']
    });

    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle2' });
    await page.evaluate(async () => {
        var header = document.getElementById('section-章节列表');
        if (header) {
            header.parentNode.removeChild(header);
        }
        var toc = document.querySelector('h1.print-page-toc-title');
        if (toc) {
            toc.parentNode.removeChild(toc);
        }
    });
    await new Promise(r => setTimeout(r, 2000));
    // sleep for a while to ensure all content is loaded
    await page.pdf({
        path: pdfPath, // path to save pdf file
        format: 'A4', // page format
        displayHeaderFooter: true, // display header and footer (in this example, required!)
        printBackground: true, // print background
        landscape: false, // use horizontal page layout
        headerTemplate: headerHtml, // indicate html template for header
        footerTemplate: footerHtml,
        scale: 1, //Scale amount must be between 0.1 and 2
        margin: { // increase margins (in this example, required!)
            top: 80,
            bottom: 80,
            left: 30,
            right: 30,
        }
    });

    await browser.close();
})();
