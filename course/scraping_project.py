import random
import requests
from bs4 import BeautifulSoup
import csv
num_of_page = 10
url = 'https://quotes.toscrape.com'
response = requests.get(url).text
for i in range(1,num_of_page):
    url = 'https://quotes.toscrape.com'
    url = url + '/page/'+str(i)+''
    response =response+requests.get(url).text

soup = BeautifulSoup(response, "html.parser")
quotes = soup.find_all(class_="quote")

url = 'https://quotes.toscrape.com'
with open("quotes_data.csv", "w") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['quote', 'author', 'bio_link'])
    for quote in quotes:
        qu = quote.find("span").get_text()
        auth = quote.find(class_="author").get_text()
        href = url + quote.find('a')['href']
        csv_writer.writerow([qu, auth, href])
        

qlist=[]
with open("quotes_data.csv")  as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        qlist.append(row)



def game(guesses=5):
    if guesses <5:
        raise ValueError("Number of guesses must be greater than 5")
    c_guesses = guesses
    r=random.randint(1,len(qlist)-1)
    author_name = qlist[r][1]
    names=author_name.split()
    hint_url=qlist[r][2]
    hint_response=requests.get(hint_url).text
    hint_soup=BeautifulSoup(hint_response, "html.parser")
    bio = hint_soup
    born_date = bio.find(class_="author-born-date").get_text()
    bio_desc = bio.find(class_="author-description").get_text()
    bio_hint=bio_desc.replace(names[0],'*****')
    bio_hint=bio_hint.replace(names[1],'*****')
    step =len(bio_hint)/(c_guesses-2)
    ran = [step*i for i in range(0,c_guesses-1)]
    print(qlist[r][0])
    while guesses>0:
        
        print(f"{guesses} guesses left")
        answer = input()
        if answer == author_name:
            print (f"You guessed right the author was {author_name}")
            guesses = 0
        else:
            if guesses == c_guesses:
                print(f"Author was born in: {born_date}")
                guesses-=1
            else:
                if guesses>2:
                    print("Here is the next hint:")
                    s1=int(ran[len(ran)-guesses])
                    s2=int(ran[len(ran)-guesses+1])
                    print(bio_hint[s1:s2])
                else:
                    s1=int(ran[len(ran)-2])
                    s2=int(ran[len(ran)-1])
                    if guesses == 2:
                        print(bio_hint[s1:s2],"\n")
                        print("It is your last guess")
                guesses-=1

        if  not guesses:    
            print("Game over\n\n")
            print("Do you want to try again? 'yes' to play again.")
            if input()=='yes':
                guesses = c_guesses
                print("New Game\n\n")
                r=random.randint(1,len(qlist)-1)
                print(qlist[r][0])
    pass
        
    
game(10)
    