from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import urllib.request
import os

# keywords
emergency_vehicle_keywords = ["救急車","警察車 パトカー","消防車"]
non_emergency_vehicle_keywords = ["日本普通の車", "日本で売れている車", "バス", "トラック", "ワゴン車"]
keywords = non_emergency_vehicle_keywords

# number of images per class
NUM_PIC = 100

# Chrome 106 version driver
driver = webdriver.Chrome('/Users/DoyoonLee/Dev/Projects/emergency-vehicle-detection/src/chromedriver')
wait = WebDriverWait(driver,5)
driver.get("https://www.google.co.kr/imghp?hl=en&tab=ri&authuser=0&ogbl")
driver.implicitly_wait(5)

for keyword in keywords:
    elem = driver.find_element(By.NAME,"q")
    COUNT=0

    # keyword with functions in Google
    elem.clear()
    elem.send_keys(f"{keyword} -おもちゃ -絵")
    elem.send_keys(Keys.RETURN)

    print("\nPrepare downloading...\n")

    # scroll down until the end
    SCROLL_PAUSE_TIME = 1
    # get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")  # 브라우저의 높이를 자바스크립트로 찾음
    while True:
        print("last_height: ",last_height)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 브라우저 끝까지 스크롤을 내림
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
            except:
                break
        last_height = new_height
    # move the window to the start
    driver.execute_script("window.scrollTo(0,0);")

    images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
    print("total number of images: ",len(images))

    print("\nStart downloading...\n")

    # prevent HTTP error 403: Forbidden
    opener=urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'MyApp/1.0')]
    urllib.request.install_opener(opener)

    if not os.path.isdir(keyword[:3]):
        os.mkdir(keyword[:3])

    for image in images:
        try:
            image.click()
            # time.sleep(1.5)
            # driver.implicitly_wait(5)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,".n3VNCb.KAlRDb")))
            imgUrl = driver.find_element(By.CSS_SELECTOR,".n3VNCb.KAlRDb").get_attribute("src")

            # download the image
            urllib.request.urlretrieve(imgUrl,f"{keyword[:3]}/{keyword[:3]+str(COUNT)}.jpg")
            print(f"{keyword[:3]}: {COUNT+1}/{len(images)}")
            COUNT+=1

            if COUNT == NUM_PIC:
                break
        except Exception as e:
            print("error: ",e)
            pass

print(f"Time Finished: {datetime.now().strftime('%H-%M-%S')}")
driver.close()