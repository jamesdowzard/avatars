const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  
  await page.goto('https://www.youtube.com/watch?v=6Dt0yCOIY2w');
  await page.waitForTimeout(5000);
  
  // Click to dismiss any dialogs
  try {
    await page.click('button[aria-label="Accept all"]', { timeout: 3000 });
  } catch (e) {}
  
  // Wait for video to load
  await page.waitForSelector('video', { timeout: 10000 });
  
  // Take screenshots at different timestamps
  const video = await page.$('video');
  const timestamps = [5, 15, 30, 45, 60, 90, 120];
  
  for (let i = 0; i < timestamps.length; i++) {
    const t = timestamps[i];
    await page.evaluate((time) => {
      const v = document.querySelector('video');
      v.currentTime = time;
    }, t);
    await page.waitForTimeout(1000);
    await page.screenshot({ 
      path: `executives/steve-butcher/frame_${t}s.png`,
      clip: { x: 0, y: 100, width: 1280, height: 720 }
    });
    console.log(`Captured frame at ${t}s`);
  }
  
  await browser.close();
})();
