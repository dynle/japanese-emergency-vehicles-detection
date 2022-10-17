from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request

# Chrome 106 version driver
driver = webdriver.Chrome('/Users/DoyoonLee/Dev/Projects/emergency-vehicle-detection/chromedriver')

emergency_vehicle_class = ["救急車","警察車","消防車"]

# for class in emergency_vehicle_class:
driver.get("https://www.google.co.kr/imghp?hl=en&tab=ri&authuser=0&ogbl")
elem = driver.find_element(By.NAME,"q")
count=0
elem.send_keys(f"{emergency_vehicle_class[0]}")
elem.send_keys(Keys.RETURN)
# images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
images = driver.find_elements(By.CSS_SELECTOR,'#islrg > div.islrc > div a.wXeWr.islib.nfEiy')
for image in images:
    try:
        image.click()
        time.sleep(2)
        # imgUrl = driver.find_element(By.XPATH,'/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute("src")
        _imgUrl = driver.find_element(By.CSS_SELECTOR, '#Sva75c > div > div > div.pxAole > div.tvh9oe.BIB1wf > c-wiz > div > div.OUZ5W > div.zjoqD > div.qdnLaf.isv-id > div > a')
        imgUrl = _imgUrl.find_element(By.TAG_NAME,'img').get_attribute('src')
        print(imgUrl)
        urllib.request.urlretrieve(imgUrl,f"{emergency_vehicle_class[0]+str(count)}"+".jpg")
        count+=1

        # Limited to num_pic
        num_pic = 3
        if count == num_pic:
            break
    except:
        pass

driver.close()