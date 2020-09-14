from Noise import *  # Importando
from Filtros import *  # Importando
import os
import numpy as np
if __name__ == '__main__':
    path = 'C:\PRUEBA'  #Cargando Imagen
    image_name = '1.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    image_grayf = image_gray.copy() #Guardando copia para visualizar
    image_gray = image_gray.astype(np.float) / 255
    image_gray_noisy = noise("s&p", image_gray.astype(np.float) )  # Generando imagen con ruido syp
    image_gray_noisy = (255 * image_gray_noisy).astype(np.uint8)
    image_gray_noisyg= noise("gauss", image_gray.astype(np.float) ) # Generando imagen con ruido gauss
    image_gray_noisyg= (255 * image_gray_noisyg).astype(np.uint8)
    filtro_ruido_syp = Filtrado(image_gray_noisy)
    filtro_ruido_gauss = Filtrado(image_gray_noisyg)
    tiempo_filtro_gauss1,ImagenFiltradaGauss1=filtro_ruido_syp.tiempo(filtro_ruido_syp.gaussian_lp) # En las siguientes linesas se mide el tiempo de ejecución de cada filtro
    tiempo_filtro_gauss2, ImagenFiltradaGauss2 = filtro_ruido_gauss.tiempo(filtro_ruido_gauss.gaussian_lp)
    tiempo_filtro_MED1, ImagenFiltradaMED1 = filtro_ruido_syp.tiempo(filtro_ruido_syp.median)
    tiempo_filtro_MED2, ImagenFiltradaMED2 = filtro_ruido_gauss.tiempo(filtro_ruido_gauss.median)
    tiempo_filtro_BILA1, ImagenFiltradaBILA1 = filtro_ruido_syp.tiempo(filtro_ruido_syp.bilateral)
    tiempo_filtro_BILA2, ImagenFiltradaBILA2 = filtro_ruido_gauss.tiempo(filtro_ruido_gauss.bilateral)
    tiempo_filtro_NLM1, ImagenFiltradaNLM1 = filtro_ruido_syp.tiempo(filtro_ruido_syp.nlm)
    tiempo_filtro_NLM2, ImagenFiltradaNLM2 = filtro_ruido_gauss.tiempo(filtro_ruido_gauss.nlm)
    print('El tiempo de ejecución del filtro Gaussiano_lp con ruido s&p es', tiempo_filtro_gauss1) #Se imprime cada tiempo medido
    print('El tiempo de ejecución del filtro Gaussiano_lp con ruido gaussiano es', tiempo_filtro_gauss2)
    print('El tiempo de ejecución del filtro Mediana con ruido s&p es', tiempo_filtro_MED1)
    print('El tiempo de ejecución del filtro Mediana con ruido gaussiano es', tiempo_filtro_MED2)
    print('El tiempo de ejecución del filtro Bilateral con ruido s&p es', tiempo_filtro_BILA1)
    print('El tiempo de ejecución del filtro Bilateral con ruido gaussiano es', tiempo_filtro_BILA2)
    print('El tiempo de ejecución del filtro Nlm con ruido s&p es', tiempo_filtro_NLM1)
    print('El tiempo de ejecución del filtro Nlm con ruido gaussiano es', tiempo_filtro_NLM2)
    print('El sqrt(ECM) de la imagen con ruido s&p y filtrada con gaussiano_lp es',filtro_ruido_syp.R_ECM(image_grayf,ImagenFiltradaGauss1)) #Calculo e impresión del ECM
    print('El sqrt(ECM) de la imagen con ruido gaussiano y filtrada con gaussiano_lp es',filtro_ruido_gauss.R_ECM(image_grayf, ImagenFiltradaGauss2))
    print('El sqrt(ECM) de la imagen con ruido s&p y filtrada con Mediana es',filtro_ruido_syp.R_ECM(image_grayf, ImagenFiltradaMED1))
    print('El sqrt(ECM) de la imagen con ruido gaussiano y filtrada con Mediana es',filtro_ruido_gauss.R_ECM(image_grayf, ImagenFiltradaMED2))
    print('El sqrt(ECM) de la imagen con ruido s&p y filtrada con Bilateral es',filtro_ruido_syp.R_ECM(image_grayf, ImagenFiltradaBILA1))
    print('El sqrt(ECM) de la imagen con ruido gaussiano y filtrada con Bilateral es',filtro_ruido_gauss.R_ECM(image_grayf, ImagenFiltradaBILA2))
    print('El sqrt(ECM) de la imagen con ruido s&p y filtrada con Nlm es',filtro_ruido_syp.R_ECM(image_grayf, ImagenFiltradaNLM1))
    print('El sqrt(ECM) de la imagen con ruido gaussiano y filtrada con Nlm es',filtro_ruido_gauss.R_ECM(image_grayf, ImagenFiltradaNLM2))
    EstimacionRuidoGauss1= filtro_ruido_syp.image_noise(image_gray_noisy,ImagenFiltradaGauss1) #Estimación de ruido para cada imagen
    EstimacionRuidoGauss2=filtro_ruido_gauss.image_noise(image_gray_noisyg,ImagenFiltradaGauss2)
    EstimacionRuidoMED1=filtro_ruido_syp.image_noise(image_gray_noisy,ImagenFiltradaMED1)
    EstimacionRuidoMED2=filtro_ruido_gauss.image_noise(image_gray_noisyg,ImagenFiltradaMED2)
    EstimacionRuidoBILA1=filtro_ruido_syp.image_noise(image_gray_noisy,ImagenFiltradaBILA1)
    EstimacionRuidoBILA2=filtro_ruido_gauss.image_noise(image_gray_noisyg,ImagenFiltradaBILA2)
    EstimacionRuidoNLM1=filtro_ruido_syp.image_noise(image_gray_noisy,ImagenFiltradaNLM1)
    EstimacionRuidoNLM2=filtro_ruido_gauss.image_noise(image_gray_noisyg,ImagenFiltradaNLM2)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoGauss1.png'), EstimacionRuidoGauss1) #Guardando las imagenes
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoGauss2.png'), EstimacionRuidoGauss2)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoMED1.png'), EstimacionRuidoMED1)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoMED2.png'), EstimacionRuidoMED2)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoBILA1.png'), EstimacionRuidoBILA1)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoBILA2.png'), EstimacionRuidoBILA2)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoNLM1.png'), EstimacionRuidoNLM1)
    # cv2.imwrite(os.path.join(path, 'EstimacionRuidoNLM2.png'), EstimacionRuidoNLM2)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaGauss1.png'), ImagenFiltradaGauss1)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaGauss2.png'), ImagenFiltradaGauss2)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaMED1.png'), ImagenFiltradaMED1)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaMED2.png'), ImagenFiltradaMED2)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaBILA1.png'), ImagenFiltradaBILA1)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaBILA2.png'), ImagenFiltradaBILA2)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaNLM1.png'), ImagenFiltradaNLM1)
    # cv2.imwrite(os.path.join(path, 'ImagenFiltradaNLM2.png'), ImagenFiltradaNLM2)
    # cv2.imwrite(os.path.join(path, 'Imagen_Ruido_1_s&p.png'), image_gray_noisy)
    # cv2.imwrite(os.path.join(path, 'Imagen_Ruido_2_gauss.png'), image_gray_noisyg)
    # cv2.imwrite(os.path.join(path, 'EscaladeGrises.png'), image_grayf)
    final_frame = cv2.hconcat((image_grayf ,ImagenFiltradaGauss1))  # Imagen sin ruido/Imagen filtrada
    cv2.imshow("image.png", final_frame)
    cv2.waitKey(0)





