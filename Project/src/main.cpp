#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <gpgpu.h>

void showFPS(GLFWwindow * window);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

constexpr int32_t kWidth = 1024;
constexpr int32_t kHeight = 1024;

char strStat[40];
int frames = 0;
char show = 0, showTexture = 1, linear = 0;
bool reset = 0;
std::chrono::time_point<std::chrono::system_clock> last = std::chrono::system_clock::now();

void main(int argc, char **argv) {
	if (!glfwInit()) {
		glfwTerminate();
	}
	//Select the OpenGL version 4.1
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	GLFWwindow* window = glfwCreateWindow(1024, 1024, "ISIMA_PROJECT", nullptr, nullptr);
	if (window) {
		//Set the OpenGL context available. OpenGL function can be called after this function
		glfwMakeContextCurrent(window);
		//Load the OpenGL function pointer from the graphic library.
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

		//Clear the background to black
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glDisable(GL_DEPTH_TEST);

		//Creation of the OpenGL texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGLSetGLDevice(0);
		cudaGraphicsResource_t cuda_graphic_resource;
		cudaGraphicsGLRegisterImage(&cuda_graphic_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		//OpenGL object to print the texture on screen
		GLuint fbo = 0;
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		//Limit the FPS to 60 (0 to set to unlimited)
		glfwSwapInterval(1);

		glfwSetKeyCallback(window, key_callback);

		//Creation of the cuda resource desc
		cudaResourceDesc cuda_resource_desc;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

		// Creation des surfaces cuda
		cudaArray_t array;
		cudaSurfaceObject_t surfaceMap = 0, surfaceRessources = 0, surfacePheromones = 0, surfaceAnts = 0;
		memset(&cuda_resource_desc, 0, sizeof(cuda_resource_desc));
		cuda_resource_desc.resType = cudaResourceTypeArray;

		// Creation de la surface associee a la carte
		cudaMallocArray(&array, &channelDesc, kWidth, kHeight, cudaArraySurfaceLoadStore);
		cuda_resource_desc.res.array.array = array;
		cudaCreateSurfaceObject(&surfaceMap, &cuda_resource_desc);

		// Creation de la surface associee aux ressources
		cudaMallocArray(&array, &channelDesc, kWidth, kHeight, cudaArraySurfaceLoadStore);
		cuda_resource_desc.res.array.array = array;
		cudaCreateSurfaceObject(&surfaceRessources, &cuda_resource_desc);

		// Creation de la surface associee aux pheromones
		cudaMallocArray(&array, &channelDesc, kWidth, kHeight, cudaArraySurfaceLoadStore);
		cuda_resource_desc.res.array.array = array;
		cudaCreateSurfaceObject(&surfacePheromones, &cuda_resource_desc);
		
		// Creation de la surface associee aux fourmis
		cudaMallocArray(&array, &channelDesc, kWidth, kHeight, cudaArraySurfaceLoadStore);
		cuda_resource_desc.res.array.array = array;
		cudaCreateSurfaceObject(&surfaceAnts, &cuda_resource_desc);

		// Initialisation du monde
		initWorld(surfaceMap, surfaceRessources, surfacePheromones, surfaceAnts);
		glfwSetKeyCallback(window, key_callback);

		// Affichage d'infos sur le GPU
		GetGPGPUInfo();
		
		FILE *fptrBackground = fopen("../src/data/Background.bmp","r");
		FILE *fptrRock = fopen("../src/data/Rock.bmp","r");
		FILE *fptrStick = fopen("../src/data/Stick.bmp","r");
		FILE *fptrWater = fopen("../src/data/Water.bmp","r");
		FILE *fptrFood = fopen("../src/data/Food.bmp","r");
		FILE *fptrAnthill = fopen("../src/data/Anthill.bmp","r");
		char sucessLoadBMPs = loadBMPs(fptrBackground,fptrRock,fptrStick,fptrWater,fptrFood,fptrAnthill);
		if(fptrBackground) fclose(fptrBackground);
		if(fptrRock) fclose(fptrRock);
		if(fptrStick) fclose(fptrStick);
		if(fptrWater) fclose(fptrWater);
		if(fptrFood) fclose(fptrFood);
		if(fptrAnthill) fclose(fptrAnthill);
		
		//Main loop
		while (!glfwWindowShouldClose(window)) {
			showFPS(window);
			glfwPollEvents();
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			//Setup the OpenGL texture 
			cudaGraphicsMapResources(1, &cuda_graphic_resource);
			cudaArray_t array_OpenGL;
			cudaGraphicsSubResourceGetMappedArray(&array_OpenGL, cuda_graphic_resource, 0, 0);
			cuda_resource_desc.res.array.array = array_OpenGL;
			cudaSurfaceObject_t surfaceAffiche;
			cudaCreateSurfaceObject(&surfaceAffiche, &cuda_resource_desc);

			//Update the simulation for each frame
			GenereImage(surfaceAffiche, surfaceMap, surfaceRessources, surfacePheromones, surfaceAnts, strStat);
			if (show) {// Affiche seulement l'ecran qu'on veut
				switch (show) {
				case 1:
					showOnly(surfaceAffiche, surfaceMap);
					break;
				case 2:
					showOnly(surfaceAffiche, surfaceRessources);
					break;
				case 3:
					showOnly(surfaceAffiche, surfacePheromones);
					break;
				case 4:
					showOnly(surfaceAffiche, surfaceAnts);
					break;
				}
			}
			
			if(sucessLoadBMPs && showTexture){
				applyTex2D(surfaceAffiche,linear);
			}
			cudaDestroySurfaceObject(surfaceAffiche);
			cudaGraphicsUnmapResources(1, &cuda_graphic_resource);
			cudaStreamSynchronize(0);

			glViewport(0, 0, width, height);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
			glBlitFramebuffer(
				0, 0, kWidth, kHeight,
				0, 0, width, height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glfwSwapBuffers(window);
			// Si il y a une demande de reset 
			if (reset) {
				printf("reset\n");
				reset = 0;
				// Destruction du monde
				destroyWorld();
				// Reconstruction du monde
				initWorld(surfaceMap, surfaceRessources, surfacePheromones, surfaceAnts);
			}
		}

		// On libere tout
		destroyWorld();
		cudaDestroySurfaceObject(surfaceMap);
		cudaDestroySurfaceObject(surfaceRessources);
		cudaDestroySurfaceObject(surfacePheromones);
		cudaDestroySurfaceObject(surfaceAnts);

		glfwDestroyWindow(window);
	}
	glfwTerminate();
}

// Affiche les FPS et les stats de la fourmiliere dans le titre de la fenetre
void showFPS(GLFWwindow * window) {
	frames++;
	std::chrono::time_point<std::chrono::system_clock> newTime = std::chrono::system_clock::now();
	std::chrono::duration<double, std::ratio<1, 1000>> elapsedseconds = newTime - last;
	if (elapsedseconds.count() > 1000) {
		char buffer[70];
		sprintf(buffer, "FPS : %2.1f %s", 1000 * (float)frames / elapsedseconds.count(),strStat); 
		glfwSetWindowTitle(window, buffer);
		last = newTime;
		frames = 0;
	}
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	static char unlimited = 1;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	// Gere l'appui sur un bouton pour changer l'ecran ou reset la simulation
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_M:
			show = 1;
			break;
		case GLFW_KEY_R:
			show = 2;
			break;
		case GLFW_KEY_P:
			show = 3;
			break;
		case GLFW_KEY_A:
			show = 4;
			break;
		case GLFW_KEY_KP_0:
		case GLFW_KEY_0:
			reset = 1;
			break;
		case GLFW_KEY_KP_ADD:
			show = (++show) % 5;
			break;
		case GLFW_KEY_KP_SUBTRACT:
			show = show ? show - 1 : 4;
			break;
		case GLFW_KEY_U:
			unlimited = !unlimited;
			glfwSwapInterval(unlimited);
			break;
		case GLFW_KEY_T:
			showTexture = !showTexture;
			break;
		case GLFW_KEY_L:
			linear = !linear;
			break;
		default:
			show = 0;
		}
	}

}
