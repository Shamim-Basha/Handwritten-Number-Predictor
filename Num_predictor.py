import pygame
import random
import tensorflow as tf
import numpy as np
import os 
from matplotlib import pyplot as plt


if not os.path.exists('predictor_model.h5'):
    dataset = tf.keras.datasets.mnist

    (x_train,y_train),(x_test,y_test) = dataset.load_data()

    x_train = tf.keras.utils.normalize(x_train,axis=1)
    x_test = tf.keras.utils.normalize(x_test,axis=1)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128,activation="relu"))
    model.add(tf.keras.layers.Dense(128,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10,activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train,y_train,epochs=10)

    model.evaluate(x_test,y_test)

    model.save('predictor_model.h5')

pygame.init()

WIDTH = 800
HEIGHT = 28*20

WHITE = "#ffffff"
BLACK = "#000000"
MODEL = tf.keras.models.load_model('predictor_model.h5')

screen = pygame.display.set_mode((WIDTH,HEIGHT))

FONT = pygame.font.SysFont("arial",30,True)

Rect = pygame.Rect(0,0,HEIGHT,HEIGHT)
pygame.draw.rect(screen,BLACK,Rect)

SIZE = 28
SQRSIZE = HEIGHT//SIZE

array = np.zeros((28,28))

def draw_grid():
    for i in range(SIZE):
        pygame.draw.line(screen,WHITE,(0,i*SQRSIZE),(HEIGHT,i*SQRSIZE))
        pygame.draw.line(screen,WHITE,(i*SQRSIZE,0),(i*SQRSIZE,HEIGHT))

def draw(x,y):
    rect = pygame.Rect(SQRSIZE*y,SQRSIZE*x,SQRSIZE,SQRSIZE)
    pygame.draw.rect(screen,WHITE,rect)
    #tl,t,tl,r,br,b,bl,l,
    neighbors = ((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
    for i,j in neighbors:
        if x+i>=0 and x+i<28 and y+j>=0 and y+j<28 and array[x+i][y+j] != 1:
            array[x+i][y+j] = random.randint(0,5)/10
    array[x][y] = 1


def erase(x,y):
    rect = pygame.Rect(SQRSIZE*y,SQRSIZE*x,SQRSIZE,SQRSIZE)
    pygame.draw.rect(screen,BLACK,rect)
    array[x][y] = 0

def draw_buttons():
    text1 = FONT.render("Reset",True,BLACK,WHITE)
    text2 = FONT.render("Predict",True,BLACK,WHITE)

    reset = pygame.Rect(HEIGHT+((WIDTH-HEIGHT)//2) - text1.get_width()//2,HEIGHT//8,text1.get_width(),text1.get_height())
    predict = pygame.Rect(HEIGHT+((WIDTH-HEIGHT)//2) - text1.get_width()//2,3*HEIGHT//8,text2.get_width(),text2.get_height())

    screen.blit(text1,(HEIGHT+((WIDTH-HEIGHT)//2) - text1.get_width()//2,HEIGHT//8))
    screen.blit(text2,(HEIGHT+((WIDTH-HEIGHT)//2) - text2.get_width()//2,3*HEIGHT//8))
    
    return reset,predict

def reset():
    global array
    array = np.zeros((28,28))
    screen.fill(BLACK,(0,0,HEIGHT,HEIGHT))
    draw_grid()

def predict():
    prediction = MODEL.predict(np.array([array]),verbose=2)[0]
    next,best = (np.argsort(prediction)[-2:])
    print(f"{best} : {prediction[best]*100}")
    print(f"{next} : {prediction[next]*100}")

draw_grid()
reset_rect,predict_rect = draw_buttons() 

running = True

while running:
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONUP:
            if reset_rect.collidepoint(pygame.mouse.get_pos()):
                reset()
            if predict_rect.collidepoint(pygame.mouse.get_pos()):
                predict()

        mouse_x,mouse_y = pygame.mouse.get_pos()
        L,_,R = pygame.mouse.get_pressed()

        if L:
            clicked_x,clicked_y = mouse_y//SQRSIZE,mouse_x//SQRSIZE
            if clicked_x>=0 and clicked_x<SIZE and clicked_y>=0 and clicked_y<SIZE:
                draw(clicked_x,clicked_y)
        if R:
            clicked_x,clicked_y = mouse_y//SQRSIZE,mouse_x//SQRSIZE
            if clicked_x>=0 and clicked_x<SIZE and clicked_y>=0 and clicked_y<SIZE:
                erase(clicked_x,clicked_y)
                draw_grid()
        

pygame.quit()
