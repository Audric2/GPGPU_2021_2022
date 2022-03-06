#include <gpgpu.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <random>

// Definition de toutes les couleurs utilisees
#define COLOR_VOID make_float4(0.f, 0.f, 0.f, 0.0f)
#define COLOR_FOOD make_float4(0.6f, 0.2f, 0.15f, 1.0f)
#define COLOR_ROCK make_float4(0.25f, 0.25f, 0.25f, 1.0f)
#define COLOR_WATER make_float4(0.0f, 0.5f, 1.0f, 1.0f)
#define COLOR_STICK make_float4(0.7f, 0.5f, 0.3f, 1.0f)
#define COLOR_BACKGROUND make_float4(0.0f, 0.5f, 0.0f, 1.0f)
#define COLOR_ANTHILL make_float4(0.28f, 0.13f, 0.f, 1.0f)

// Definition de toutes les couleurs pour les fourmis
#define COLOR_ANT_SEARCH make_float4(1.f, 1.f, 1.f, 1.0f)
#define COLOR_ANT_SEARCH_SHOW make_float4(0.f, 0.f, 0.f, 1.0f)
#define COLOR_ANT_WATER make_float4(0.f, 0.5f, 1.0f, 1.0f)
#define COLOR_ANT_WATER_SHOW make_float4(0.f, 0.f, 1.0f, 1.0f)
#define COLOR_ANT_FOOD make_float4(0.9f, 0.3f, 0.23f, 1.0f)
#define COLOR_ANT_FOOD_SHOW make_float4(1.f, 0.f, 0.f, 1.0f)

// Definition des types de comportements
#define ANT_SEARCH 1
#define ANT_WATER 2
#define ANT_FOOD 4

// Parametres de la dispersion des pheromones
// pour la pose de pheromones
#define ANT_SEARCH_WEIGHT 0.3f
#define ANT_WATER_WEIGHT 0.4f
#define ANT_FOOD_WEIGHT 0.4f
// pour la dispersion des pheromones
#define ANT_SEARCH_DISPERSE 0.0001f
#define ANT_WATER_DISPERSE 0.002f
#define ANT_FOOD_DISPERSE 0.002f
// pour la dilution des pheromones
#define DILLUTION_RAD 2

// Pi
constexpr float PI = 3.141592653589793f;
// Taille des pixels en espace 1 par 1
constexpr float SIZE_PIXEL = (1.f / 1024.0f);
// Rayon de recherche des fourmis
constexpr float ANT_RAD_SEARCH = 9.f * SIZE_PIXEL;
// Pour tous les parametres suivants : le 1 devant signifie qu'ils sont actifs et 0 inactifs
// Si on met un 0 a la place de tous les 1 on obtient des fourmis qui se comportent comme des lancers de rayons  
// Pheromone de base de la fourmiliere
constexpr float WEIGHTAnthillBase = 0 * 4096.f;
// Poids ajoute lorsque qu'une pheromone est trouvee
constexpr float WEIGHTForPheromones = 1 * 0.2f * (2 * ANT_RAD_SEARCH / SIZE_PIXEL + 1) * (2 * ANT_RAD_SEARCH / SIZE_PIXEL + 1);
// Poids de base d'une case pour valoir plus qu'une case indisponible
constexpr float BaseWEIGHT = 1 * 1.f;
// Poids ajoute lorsque qu'une fourmi trouve son but
constexpr float WEIGHTGoal = 1 * 500000.f;
// Taille de la moitie du carre de recherche
constexpr float midRad = 0.5f * (2 * ANT_RAD_SEARCH / SIZE_PIXEL + 1) * (2 * ANT_RAD_SEARCH / SIZE_PIXEL + 1);
// Angle maximum pris lors d'un choc avec un rocher
constexpr float AngleChocRock = 1 * PI / 2.f;
// Angle maximum pris lorsque la fourmi avance tout droit
constexpr float deltaAngleChemin = 1 * PI / 10.f;
// Angle maximum pris lors d'un choc avec une fourmi
constexpr float AngleChocFourmi = 1 * PI / 6.f;
// Position maximum pris lors d'un choc avec une fourmi
constexpr float deltaPosChocFourmi = 1 * 1 * SIZE_PIXEL;
// Perte de la puissance de pheromone par frame
constexpr float lostWeightPerFrame = 1.f/(1024.f);
// Tres grande valeur utilisee pour faire des test de compraison
constexpr float InfinityF = HUGE_VALF;

// Definition d'un rocher
typedef struct Rock {
	float2 p;
	float radius;
} Rock;

// Definition d'un baton
typedef struct Stick {
	float2 p;
	float2 dim;
	float ang;
} Stick;

// Definition d'une ressource eau
typedef struct Water {
	float2 p;
	float radius;
} Water;

// Definition d'une ressource nourriture
typedef struct Food {
	float2 p;
	float radius;
} Food;

// Definition d'une fourmiliere
typedef struct Anthill {
	float2 p;
	float radius;
	int food;
	int water;
} Anthill;

// Definition d'une fourmi
typedef struct Ant {
	float2 p;
	float direction;
	float pheromonePower;
	int type;
} Ant;

// Definition des tableaux des diffferentes structures 
int nbRocks;
Rock * device_rocks = nullptr;
int nbSticks;
Stick * device_sticks = nullptr;
int nbWaters;
Water * device_waters = nullptr;
int nbFoods;
Food * device_foods = nullptr;
int nbAnthills;
Anthill * device_anthills = nullptr;
int nbAnts;
Ant * device_ants = nullptr;



// Affiche des proprietes du GPU
void GetGPGPUInfo() {
	cudaDeviceProp cuda_propeties;
	cudaGetDeviceProperties(&cuda_propeties, 0);
	int multiProcessorCount = cuda_propeties.multiProcessorCount;
	int maxBlockPerMultiProcessor = cuda_propeties.maxBlocksPerMultiProcessor;
	int maxThreadPerBlock = cuda_propeties.maxThreadsPerBlock;
	int maxThreadPerMultiProcessor  = cuda_propeties.maxThreadsPerMultiProcessor;
	printf("Multi-Processor = %d\n", multiProcessorCount);
	printf("Nb max de Block par Multi-Processor = %d\n", maxBlockPerMultiProcessor);
	printf("Nb max de Threads par Block = %d\n", maxThreadPerBlock);
	printf("Nb max de Threads par Multi-Processor = %d\n",maxThreadPerMultiProcessor);
}

// Recuperation de la partie non entiere de x
__device__ float fracf(float x){
	return x - floorf(x);
}

// Modulo etendu au float
__device__ float modf(float x,float y){
	return x - floorf(x/y)*y;
}

// Genere un nombre "aleatoire" entre 0 et 1 [0;1[
__device__ float random(float x, float y) {
	float t = 12.9898f * x + 78.233f * y;
	return abs(fracf(t * sin(t)));
}

// Definition de differents operateurs pour les float2 et float4
__device__ float2 operator-(float2 &a, float2 &b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ float4 operator+(float4 &a, float4 &b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);}
__device__ float4 operator*(float4 &a, float4 &b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);}
__device__ float4 operator*(float4 &a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);}
__device__ bool operator==(float4 &a, float4 &b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;}
__device__ bool operator!=(float4 &a, float4 &b) { return !(a == b);}

// Calcule la distance entre les coordonnees p et une boite orientee centree en (0,0), de dimension dim et d'orientation angle
__device__ float OrientedBox(float2 p, float2 dimBox, float angle) {
	float2 w = make_float2(cos(angle), sin(angle));
	float4 q = make_float4(w.x, w.y, w.y, w.x) * make_float4(p.x, p.x, p.y, p.y);
	float4 s = make_float4(w.x, w.y, w.y, w.x) * make_float4(dimBox.x, dimBox.x, dimBox.y, dimBox.y);
	return max(	max(abs(q.x + q.z) - dimBox.x, abs(q.w - q.y) - dimBox.y) /max(abs(w.x - w.y), abs(w.x + w.y)),
		max(abs(p.x) - max(abs(s.x - s.z), abs(s.x + s.z)),abs(p.y) - max(abs(s.y + s.w), abs(s.y - s.w))));
}

// Borne un float entre 0 et 1 inclus
__device__ float borne(float f) {
	return min(max(f,0.f),1.f);
}

/** Calcule la direction que la fourmi en index va prendre, basee sur une ponderation des cases devant elle
* 	Le boolean de retour indique si oui ou non la fourmi devrait avancer
*/
__device__  bool getDirectionWeight(cudaSurfaceObject_t surfaceMapRessources, cudaSurfaceObject_t surfacePheromones, uint32_t width, uint32_t height, int index, Ant* ant) {
	float x = 0, y = 0, n = 0, weight;
	int nbAccesible = 0;
	float4 colorpix, colorPheromones;
	const float centerX = (ant->p.x + cos(ant->direction) * (ANT_RAD_SEARCH + SIZE_PIXEL));
	const float centerY = (ant->p.y + sin(ant->direction) * (ANT_RAD_SEARCH + SIZE_PIXEL));
	const float minRange = -1 * ANT_RAD_SEARCH;
	const float maxRange = ANT_RAD_SEARCH + SIZE_PIXEL;
	// On regarde le carre devant nous centre en (centerX,centerY) et de "rayon" ANT_RAD_SEARCH 
	for (float j = minRange; j < maxRange; j += SIZE_PIXEL) {
		for (float i = minRange; i < maxRange; i += SIZE_PIXEL) {
			const float u = centerX + i;
			const float v = centerY + j;
			if (0 <= u && u <= 1 && 0 <= v && v <= 1) { // Si la coordonnee est sur l'image 
				const int pX = (uint32_t)(u * (float)(width - 1));
				const int pY = (uint32_t)(v * (float)(height - 1));
				surf2Dread(&colorpix, surfaceMapRessources, pX * sizeof(float4), pY);
				// On verifie que le pixel lu n'est pas un rocher
				if (colorpix != COLOR_ROCK) {
					surf2Dread(&colorPheromones, surfacePheromones, pX * sizeof(float4), pY);
					// On donne un poids de base pour eviter qu'une case libre ait le meme poids qu'une case inaccessible
					weight = BaseWEIGHT;
					// On ajoute un poids a la case si elle a les pheromones recherchees par la fourmi ou si c'est le but de la fourmi
					if (ant->type == ANT_SEARCH) {
						if (colorpix == COLOR_FOOD || colorpix == COLOR_WATER) weight += WEIGHTGoal;
						else weight += max(colorPheromones.x, colorPheromones.z) * WEIGHTForPheromones;
					}
					else {
						if (colorpix == COLOR_ANTHILL) weight += WEIGHTGoal;
						else weight += colorPheromones.y * WEIGHTForPheromones;
					}
					// On ajoute les coordonnees de la case en la ponderant
					x += (float)i * (float)weight;
					y += (float)j * (float)weight;
					n += (float)weight;
					nbAccesible++;
				}
			}
		}
	}
	// La majorite des cases devant moi sont inacessibles et il n'y a pas de pheromones dans la zone
	if (n <= BaseWEIGHT*midRad && nbAccesible < midRad) {
		// On fait demi-tour en ajoutant un angle de choc aleatoire
		ant->direction += PI + (random((index + 1) * ant->p.x, (index + 1) * ant->p.y) - 0.5f) * 2 * AngleChocRock;
		return false;
	}
	// Si n = 0 (possible si on desactive le poids de base) on le met a 1 pour eviter /0
	if (!n) n = 1;
	// Case que la fourmi visera
	const float finalX = centerX + (x / n);
	const float finalY = centerY + (y / n);
	// Calcul de la direction en fonction de la case visee
	ant->direction = atan2(finalY - ant->p.y, finalX - ant->p.x);
	return true;
}

// Calcule la direction que la fourmi en index va prendre, basee sur la case avec la valeur de pheromone la plus grande
// Plus utilisee
__device__  bool getDirectionMax(cudaSurfaceObject_t surfaceMapRessources, cudaSurfaceObject_t surfacePheromones, uint32_t width, uint32_t height, int index, Ant* ant) {
	float x = 0, y = 0;
	const float posX = (ant->p.x + cos(ant->direction) * (ANT_RAD_SEARCH + SIZE_PIXEL));
	const float posY = (ant->p.y + sin(ant->direction) * (ANT_RAD_SEARCH + SIZE_PIXEL));
	const float minRange = -1 * ANT_RAD_SEARCH;
	const float maxRange = ANT_RAD_SEARCH + SIZE_PIXEL;
	int nbX = 0;
	int nbGoal = 0;
	float maxPheromone = 0.f;
	float4 colorPheromone;
	for (float j = minRange; j < maxRange; j += SIZE_PIXEL) {
		for (float i = minRange; i < maxRange; i += SIZE_PIXEL) {
			const float u = posX + i;
			const float v = posY + j;
			if (0 <= u && u <= 1 && 0 <= v && v <= 1) {// Si la coordonnee est sur l'image 
				float4 colorpix;
				const int pX = (uint32_t)(u * (float)(width - 1));
				const int pY = (uint32_t)(v * (float)(height - 1));
				surf2Dread(&colorpix, surfaceMapRessources, pX * sizeof(float4), pY);
				surf2Dread(&colorPheromone, surfacePheromones, pX * sizeof(float4), pY);
				nbX += (colorpix == COLOR_ROCK);
				if (pX != ant->p.x * (width - 1) || pY != ant->p.y * (height - 1)) {
					if (ant->type == ANT_SEARCH) {
						if (colorpix == COLOR_FOOD || colorpix == COLOR_WATER) {
							if (maxPheromone == InfinityF) {
								x = x + i;
								y = y + j;
							}
							else {
								x = i;
								y = j;
							}
							nbGoal++;
							maxPheromone = InfinityF;
						}
						else {
							if (max(colorPheromone.x, colorPheromone.z) > maxPheromone) {
								x = i;
								y = j;
								maxPheromone = max(colorPheromone.x, colorPheromone.z);
							}
						}
					}else {
						if (colorpix == COLOR_ANTHILL) {
							if (maxPheromone == InfinityF) {
								x = x + i;
								y = y + j;
							} else {
								x = i;
								y = j;
							}
							nbGoal++;
							maxPheromone = InfinityF;
						} else {
							if (colorPheromone.y > maxPheromone) {
								x = i;
								y = j;
								maxPheromone = colorPheromone.y;
							}
						}
					}
				}
				else nbX++;
			}
		}
	}
	// La majorite des cases devant moi sont inacessible
	if (nbX > midRad) {
		ant->direction += PI + (random((index + 1) * ant->p.x, (index + 1) * ant->p.y) - 0.5f) * 2 * AngleChocRock;
		return false;
	}
	// J'ai le but devant moi
	if (nbGoal){
		x /= nbGoal;
		y /= nbGoal;
	}
	const float finalX = posX + x;
	const float finalY = posY + y;
	ant->direction = atan2(finalY - ant->p.y, finalX - ant->p.x);
	return true;
}

// Fonction de premiere ecriture de la surFacePheromones
__global__  void kernel_init_draw_pheromone(cudaSurfaceObject_t surfacePheromones, cudaSurfaceObject_t surfaceRessources, uint32_t width, uint32_t height) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorpix, colorPheromone = COLOR_VOID;
	surf2Dread(&colorpix, surfaceRessources, x * sizeof(float4), y);
	if (colorpix == COLOR_ANTHILL) colorPheromone.y = WEIGHTAnthillBase;
	surf2Dwrite(colorPheromone, surfacePheromones, x * sizeof(float4), y);
}

// Fonction de premiere ecriture de la surfaceAnts
__global__  void kernel_init_draw_ants(cudaSurfaceObject_t surfaceAnts, uint32_t width, uint32_t height) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	surf2Dwrite(COLOR_VOID, surfaceAnts, x * sizeof(float4), y);
}

// Fonction de premiere ecriture de la surfaceMap
__global__  void kernel_init_draw_map(cudaSurfaceObject_t surfaceMap, uint32_t width, uint32_t height, Rock* rocks, int nbRocks, Stick* sticks, int nbSticks) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float xf = (float)x / (float)(width-1);
	float yf = (float)y / (float)(height-1);
	float2 coordPix = make_float2(xf, yf);

	for (int i = 0; i < nbRocks; i++) {	
		if (hypotf(rocks[i].p.x - xf, rocks[i].p.y - yf) < rocks[i].radius) {
			surf2Dwrite(COLOR_ROCK, surfaceMap, x * sizeof(float4), y);
			return;
		}
	}
	for (int i = 0; i < nbSticks; i++) {
		if (OrientedBox(sticks[i].p - coordPix, sticks[i].dim, sticks[i].ang) < 0) {
			surf2Dwrite(COLOR_STICK, surfaceMap, x * sizeof(float4), y);
			return;
		}
	}
	surf2Dwrite(COLOR_BACKGROUND, surfaceMap, x * sizeof(float4), y);
}

// Fonction de premiere ecriture de la surfaceRessource
__global__  void kernel_init_draw_ressources(cudaSurfaceObject_t surfaceRessource, cudaSurfaceObject_t surfaceMap, uint32_t width, uint32_t height, Water* waters, int nbWaters, Food* foods, int nbFoods, Anthill* anthills, int nbAnthills) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorMap;
	float xf = (float)x / (float)width;
	float yf = (float)y / (float)height;

	for (int i = 0; i < nbAnthills; i++) {
		if (hypotf(anthills[i].p.x - xf, anthills[i].p.y - yf) < anthills[i].radius) {
			surf2Dwrite(COLOR_ANTHILL, surfaceRessource, x * sizeof(float4), y);
			return;
		}
	}
	surf2Dread(&colorMap, surfaceMap, x * sizeof(float4), y);
	if (colorMap == COLOR_ROCK) {
		surf2Dwrite(COLOR_VOID, surfaceRessource, x * sizeof(float4), y);
		return;
	}
	
	for (int i = 0; i < nbWaters; i++) {
		if (hypotf(waters[i].p.x - xf, waters[i].p.y - yf) < waters[i].radius) {
			surf2Dwrite(COLOR_WATER, surfaceRessource, x * sizeof(float4), y);
			return;
		}
	}

	for (int i = 0; i < nbFoods; i++) {
		if (hypotf(foods[i].p.x - xf, foods[i].p.y - yf) < foods[i].radius) {
			surf2Dwrite(COLOR_FOOD, surfaceRessource, x * sizeof(float4), y);
			return;
		}
	}
	surf2Dwrite(COLOR_VOID, surfaceRessource, x * sizeof(float4), y);
}

// Met la fusion de la surfaceMap et de surfaceRessources dans la surfaceOut
__global__  void kernel_fuse_map(cudaSurfaceObject_t surfaceOut, cudaSurfaceObject_t surfaceMap, cudaSurfaceObject_t surfaceRessources, uint32_t width, uint32_t height) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorpix;
	surf2Dread(&colorpix, surfaceRessources, x * sizeof(float4), y);
	if (!colorpix.x && !colorpix.y && !colorpix.z) {
		surf2Dread(&colorpix, surfaceMap, x * sizeof(float4), y);
	}
	surf2Dwrite(colorpix, surfaceOut, x * sizeof(float4), y);
}

// Copie la surfaceIn dans la surfaceOut
__global__  void kernel_copy_map(cudaSurfaceObject_t surfaceOut, cudaSurfaceObject_t surfaceIn, uint32_t width, uint32_t height) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorpix;
	surf2Dread(&colorpix, surfaceIn, x * sizeof(float4), y);
	surf2Dwrite(colorpix, surfaceOut, x * sizeof(float4), y);
}

// Gere l'affichage et le comportement d'une fourmi
__global__  void kernel_draw_ant(cudaSurfaceObject_t surfaceMapRessources, cudaSurfaceObject_t surfaceRessources, cudaSurfaceObject_t surfacePheromones, cudaSurfaceObject_t surfaceAnts, uint32_t width, uint32_t height, Anthill* anthills, int nbAnthills, Ant * ants, int nbAnts) {

	int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	// La fourmi recupere sa nouvelle direction et si elle doit avancer a cette frame ou non
	const bool b = getDirectionWeight(surfaceMapRessources, surfacePheromones, width, height, index, &ants[index]);
	if (!b) {
		return;
	}
	int32_t lx = ants[index].p.x * (width - 1);
	int32_t ly = ants[index].p.y * (height - 1);
		
	// la fourmi avance
	ants[index].p.x = borne(ants[index].p.x + cos(ants[index].direction)*SIZE_PIXEL);
	ants[index].p.y = borne(ants[index].p.y + sin(ants[index].direction)*SIZE_PIXEL);

	// La fourmi change sa direction legerement 
	ants[index].direction += (random((index + 1) * ants[index].p.x, (index + 1) * ants[index].p.y) - 0.5f) * deltaAngleChemin * 2.f;
	ants[index].pheromonePower = max(0.f, ants[index].pheromonePower-lostWeightPerFrame);

	int32_t x = ants[index].p.x * (width - 1);
	int32_t y = ants[index].p.y * (height - 1);

	float4 colorpix, colorpheromone, colorant;
	surf2Dread(&colorpix, surfaceMapRessources, x * sizeof(float4), y);
	surf2Dread(&colorant, surfaceAnts, x * sizeof(float4), y);

	bool supprRessource = false;
	// Collision avec une ancienne fourmi
	if (colorant != COLOR_VOID) {
		ants[index].p.x += (random((index + 1) * ants[index].p.x, (index + 1) * ants[index].p.y) - 0.5f) * 2 * deltaPosChocFourmi;
		ants[index].p.y += (random((index + 1) * ants[index].p.x, (index + 1) * ants[index].p.y) - 0.5f) * 2 * deltaPosChocFourmi;
		x = ants[index].p.x * (width - 1);
		y = ants[index].p.y * (height - 1);
		ants[index].direction += (random((index + 1) * ants[index].p.x, (index + 1) * ants[index].p.y) - 0.5f) * 2 * AngleChocFourmi;
	}

	// Si la fourmi a trouve son but elle change de type, fait demi-tour et remet a 1 la puissance de ses pheromones
	if (ants[index].type == ANT_SEARCH && colorpix == COLOR_FOOD) {
		ants[index].type = ANT_FOOD;
		ants[index].direction += PI;
		/** Indique qu'il faudra supprimer le pixel de nourriture de surfacesRessources 
		* 	La nature de la suppresion fait que 2 fourmi peuvent prendre la meme ressource si elles:
		*		- sont au meme endroit a la meme frame 
		*		- cherchent le meme type de ressources
		*/
		supprRessource = true;
		ants[index].pheromonePower = 1.f;
	}else if (ants[index].type == ANT_SEARCH && colorpix == COLOR_WATER) {
		ants[index].type = ANT_WATER;
		ants[index].direction += PI;
		ants[index].pheromonePower = 1.f;
	}else if(ants[index].type != ANT_SEARCH && colorpix == COLOR_ANTHILL) {
		/** Certaines ecritures ne sont pas protegees par des semaphores, ne sont pas atomiques 
		*	et n'etaient pas protegeable simplement avec un __syncthread
		*	donc dans la suite du code il y a des ecritures/lecture concurrentes entre les threads qui ne sont pas gerees :
		*	 - lors de l'augmentation de l'eau de la fourmiliere
		*	 - lors de l'augmentation de la nourriture de la fourmiliere
		*	donc si 2+ fourmis ramenent la meme ressource a la meme frame, le comportement n'est pas defini
		*/ 
		if (ants[index].type == ANT_WATER) anthills[0].water++;
		if (ants[index].type == ANT_FOOD) anthills[0].food++;
		ants[index].type = ANT_SEARCH;
		ants[index].direction += PI;
		ants[index].pheromonePower = 1.f;
	}

	// fixe la couleur a afficher en fonction du type de fourmi
	switch (ants[index].type) {
	case ANT_SEARCH:
		colorpix = COLOR_ANT_SEARCH_SHOW;
		colorant = COLOR_ANT_SEARCH;
		break;
	case ANT_WATER:
		colorpix = COLOR_ANT_WATER_SHOW;
		colorant = COLOR_ANT_WATER;
		break;
	case ANT_FOOD:
		colorpix = COLOR_ANT_FOOD_SHOW;
		colorant = COLOR_ANT_FOOD;
		break;
	}
	
	// On augmente les pheromones sur la case ou l'on est 
	surf2Dread(&colorpheromone, surfacePheromones, x * sizeof(float4), y);	
	switch (ants[index].type) {
	case ANT_SEARCH:
		colorpheromone.y += ANT_SEARCH_WEIGHT * ants[index].pheromonePower;
		break;
	case ANT_WATER:
		colorpheromone.z += ANT_WATER_WEIGHT * ants[index].pheromonePower;
		break;
	case ANT_FOOD:
		colorpheromone.x += ANT_FOOD_WEIGHT * ants[index].pheromonePower;
		break;
	}
	/** On s'assure que les ecritures se font apres toutes les lectures pour eviter des acces concurents
	* 	Il reste tout de meme des problemes dans le cas ou plusieurs fourmis sont au meme endroit car les ecritures seront concurrentes :
	*		- si plusieurs type de fourmis sont au meme endroit, on ne sait pas laquelles ecrira en dernier sa couleur dans surfaceMapRessources,surfaceAnts
	*		- les pheromones n'augmenteront que de 1 fourmi (la derniere a ecrire) et le type aussi ne changera que de 1 dans surfacePheromones
	*/
	__syncthreads();
	surf2Dwrite(COLOR_VOID, surfaceAnts, lx * sizeof(float4), ly);
	__syncthreads();
	surf2Dwrite(colorpix, surfaceMapRessources, x * sizeof(float4), y);
	surf2Dwrite(colorant, surfaceAnts, x * sizeof(float4), y);
	if(supprRessource){
		surf2Dwrite(COLOR_VOID, surfaceRessources, x * sizeof(float4), y);
	}
	surf2Dwrite(colorpheromone, surfacePheromones, x * sizeof(float4), y);
	
}

// Supprime une partie des pheromones a chaque frame
__global__  void kernel_disperse_pheromones(cudaSurfaceObject_t surfacePheromones, uint32_t width, uint32_t height) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorFinal = COLOR_VOID;
	surf2Dread(&colorFinal, surfacePheromones, x * sizeof(float4), y);
	colorFinal.x = max(colorFinal.x - ANT_FOOD_DISPERSE, 0.f);
	colorFinal.y = max(colorFinal.y - ANT_SEARCH_DISPERSE, 0.f);
	colorFinal.z = max(colorFinal.z - ANT_WATER_DISPERSE, 0.f);
	__syncthreads();
	surf2Dwrite(colorFinal, surfacePheromones, x * sizeof(float4), y);
}

// Dilue une partie des pheromones a chaque frame
__global__  void kernel_dillution_pheromones(cudaSurfaceObject_t surfacePheromones, uint32_t width, uint32_t height) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorTmp, colorFinal = COLOR_VOID;
	float n = 0;
	for (int dx = -DILLUTION_RAD; dx <= DILLUTION_RAD; dx++) {
		for (int dy = -DILLUTION_RAD; dy <= DILLUTION_RAD; dy++) {
			if (0 <= x + dx && x + dx < width && 0 <= y + dy && y + dy < height) {
				surf2Dread(&colorTmp, surfacePheromones, (x + dx) * sizeof(float4), y + dy);
				const float weight = expf((float)((1 + DILLUTION_RAD - abs(dx)) * (1 + DILLUTION_RAD - abs(dy))));
				colorFinal = colorFinal + colorTmp * weight;
				n += weight;
			}
		}
	}
	if (n)
		colorFinal = colorFinal * (1 / n);
	// On attend que toutes les lectures aient ete faites avant d'ecrire
	__syncthreads();
	surf2Dwrite(colorFinal, surfacePheromones, x * sizeof(float4), y);
}

// Trouve une bonne position pour placer une ressource de type Food en evitant les rochers
__device__ float2 findGoodPosFood(Food &food, cudaSurfaceObject_t surfaceMap, uint32_t width, uint32_t height) {
	float2 p = make_float2(food.p.x, food.p.y);
	float4 colormap = COLOR_VOID;
	bool possible = false;
	while (!possible) {
		// Trouve une position au hasard
		p.x = random(p.x, p.y);
		p.y = random(p.x, p.y);
		const uint32_t x = p.x * (width - 1),y = p.y * (height - 1);
		possible = true;
		surf2Dread(&colormap, surfaceMap, x * sizeof(float4), y);
		// Verifie que la position n'est pas dans un caillou
		if (colormap == COLOR_ROCK) {
			possible = false;
		}
	}
	return p;
}

// Verifie et regenere les ressources de type Food qui en ont besoin
__global__  void kernel_regen_ressources(cudaSurfaceObject_t surfaceRessources, cudaSurfaceObject_t surfaceMap, uint32_t width, uint32_t height, Water* waters, int nbWaters, Food* foods, int nbFoods, Rock* rocks, int nbRocks) {
	int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	int isFood = 0;
	// On considere qu'un Food est mange a partir du moments ou certain points ont étés mangés
	for (float j = -1; j <= 1; ++j) {
		for (float i = -1; i <= 1; ++i) {
			const float u = foods[index].p.x + i * foods[index].radius / 2.f;
			const float v = foods[index].p.y + j * foods[index].radius / 2.f;
			if (0 <= u && u <= 1 && 0 <= v && v <= 1) {
				float4 colorpix;
				const int pX = (uint32_t)(u * (float)(width - 1));
				const int pY = (uint32_t)(v * (float)(height - 1));
				surf2Dread(&colorpix, surfaceRessources, pX * sizeof(float4), pY);
				if (colorpix == COLOR_FOOD) {
					isFood++;
				}
			}
		}
	}
	// Verifie si le Food doit etre regenere
	if (isFood <= 2) {
		// Redessine le food sur la surface des ressources
		foods[index].p = findGoodPosFood(foods[index], surfaceMap, width, height);
		foods[index].radius = random(foods[index].p.x, foods[index].p.y) * 0.01f + 0.01f;
		const float startX = (foods[index].p.x - foods[index].radius),
			startY = (foods[index].p.y - foods[index].radius),
			endX = (foods[index].p.x + foods[index].radius),
			endY = (foods[index].p.y + foods[index].radius);
		float4 colormap = COLOR_VOID, colorressource = COLOR_VOID;
		for (float xf = startX; xf <= endX; xf += SIZE_PIXEL) {
			for (float yf = startY; yf <= endY; yf += SIZE_PIXEL) {
				const int x = (uint32_t)(min(max(xf, 0.f), 1.f) * (float)(width - 1));
				const int y = (uint32_t)(min(max(yf, 0.f), 1.f) * (float)(height - 1));
				surf2Dread(&colormap, surfaceMap, x * sizeof(float4), y);
				surf2Dread(&colorressource, surfaceRessources, x * sizeof(float4), y);
				// On verifie que le pixel est sur l'ecran, n'est pas deja un caillou et qu'il n'y a pas deja une ressource
				if (0 <= xf && xf <= 1 && 0 <= yf && yf <= 1 && hypotf(foods[index].p.x - xf, foods[index].p.y - yf) < foods[index].radius && colormap!=COLOR_ROCK && colorressource==COLOR_VOID) {
					surf2Dwrite(COLOR_FOOD, surfaceRessources, x * sizeof(float4), y);
				}
			}
		}
	}
}

// Trouve une position qui n'est pas deja occupe par un caillou a partir de la distribution donnee
float2 findGoodPos(std::uniform_real_distribution<>& disCoord, std::mt19937& gen, Rock* rocks, int nbRocks) {
	float2 p = make_float2(0.f, 0.f);
	bool possible = false;
	while (!possible) {
		p.x = (float)disCoord(gen);
		p.y = (float)disCoord(gen);
		possible = true;
		int i = 0;
		while (i < nbRocks) {
			if (hypotf(rocks[i].p.x - p.x, rocks[i].p.y - p.y) < rocks[i].radius) {
				possible = false;
			}
			++i;
		}
	}
	return p;
}

// Trouve une position sur l'exterieur de l'ecran qui n'est pas deja occupee par un caillou a partir de la distribution donnee
float2 findGoodPosForExt(std::uniform_real_distribution<>& disCoord, std::mt19937& gen, Rock* rocks, int nbRocks,float bound) {
	float2 p = make_float2(0.f, 0.f);
	bool possible = false;
	while (!possible) {
		p.x = (float)disCoord(gen);
		p.y = (float)disCoord(gen);
		// on verifie que la position n'est pas au milieu l'ecran en utilisant bound comme seuil
		possible = !(bound < p.x && p.x < 1 - bound && bound < p.y&& p.y < 1 - bound);
		int i = 0;
		while (possible && i < nbRocks) {
			if (hypotf(rocks[i].p.x - p.x, rocks[i].p.y - p.y) < rocks[i].radius) {
				possible = false;
			}
			++i;
		}
	}
	return p;
}

// Initialise la totalite du monde
void initWorld(cudaSurfaceObject_t surfaceMap, cudaSurfaceObject_t surfaceRessources, cudaSurfaceObject_t surfacePheromones, cudaSurfaceObject_t surfaceAnts) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> disNbStick(1000.0, 2000.1);
	std::uniform_real_distribution<> disNbRock(20.0, 50.1);//42
	std::uniform_real_distribution<> disNbWater(2.0, 6.1);//3
	std::uniform_real_distribution<> disNbFood(10.0, 20.1);//15

	std::uniform_real_distribution<> disCoord(0.0, 1.0);
	std::uniform_real_distribution<> disCoordAnthill(0.4, 0.6);

	std::uniform_real_distribution<> disRadRock(0.01, 0.04);
	std::uniform_real_distribution<> disDimStick(0.0001, 0.01);
	std::uniform_real_distribution<> disRadWater(0.03, 0.1);
	std::uniform_real_distribution<> disRadFood(0.01, 0.03);
	std::uniform_real_distribution<> disRadAnthill(0.02, 0.03);

	std::uniform_real_distribution<> disAnt(-0.02, 0.02);

	std::uniform_real_distribution<> disDeg(0.0, 2 * PI);

	nbRocks = (int)disNbRock(gen);
	nbSticks = (int)disNbStick(gen);
	nbWaters = (int)disNbWater(gen);
	nbFoods = (int)disNbFood(gen);
	nbAnthills = 1;
	nbAnts = 4*1024;
	// Allocation de la memoire pour creer les tableaux des diffferentes structures
	Rock * host_rocks = (Rock*)malloc(nbRocks * sizeof(Rock));
	Stick * host_sticks = (Stick*)malloc(nbSticks * sizeof(Stick));
	Water * host_waters = (Water*)malloc(nbWaters * sizeof(Water));
	Food * host_foods = (Food*)malloc(nbFoods * sizeof(Food));
	Anthill * host_anthills = (Anthill*)malloc(nbAnthills * sizeof(Anthill));
	Ant * host_ants = (Ant*)malloc(nbAnts * sizeof(Ant));
	// Initialisation de toutes les structures
	for (int i = 0; i < nbRocks; ++i) {
		host_rocks[i].p.x = (float)disCoord(gen);
		host_rocks[i].p.y = (float)disCoord(gen);
		host_rocks[i].radius = (float)(disRadRock(gen));
	}
	for (int i = 0; i < nbSticks; ++i) {
		host_sticks[i].p.x = (float)disCoord(gen);
		host_sticks[i].p.y = (float)disCoord(gen);
		host_sticks[i].dim.x = (float)disDimStick(gen);
		host_sticks[i].dim.y = (float)disDimStick(gen) / 2.f;
		host_sticks[i].ang = (float)disDeg(gen);
	}
	for (int i = 0; i < nbWaters; ++i) {
		host_waters[i].p = findGoodPosForExt(disCoord, gen, host_rocks, nbRocks, 0.05);
		host_waters[i].radius = (float)(disRadWater(gen));
	}
	for (int i = 0; i < nbFoods; ++i) {
		host_foods[i].p = findGoodPos(disCoord, gen, host_rocks, nbRocks);
		host_foods[i].radius = (float)(disRadFood(gen));
	}
	for (int i = 0; i < 1; ++i) {
		host_anthills[i].p = findGoodPos(disCoordAnthill, gen, host_rocks, nbRocks);
		host_anthills[i].radius = (float)(disRadAnthill(gen));
		host_anthills[i].water = 0;
		host_anthills[i].food = 0;
	}
	for (int i = 0; i < nbAnts; ++i) {
		host_ants[i].direction = disDeg(gen);
		host_ants[i].p.x = host_anthills[0].p.x;// +cos(host_ants[i].direction) * disAnt(gen);
		host_ants[i].p.y = host_anthills[0].p.y;// +sin(host_ants[i].direction) * disAnt(gen);
		host_ants[i].pheromonePower = 1.f;
		host_ants[i].type = ANT_SEARCH;
	}


	// copie les tableaux de l'host a la carte graphique
	cudaMalloc(&device_rocks, nbRocks * sizeof(Rock));
	cudaMemcpy(
		device_rocks,
		host_rocks,
		nbRocks * sizeof(Rock),
		cudaMemcpyHostToDevice);
	cudaMalloc(&device_sticks, nbSticks * sizeof(Stick));
	cudaMemcpy(
		device_sticks,
		host_sticks,
		nbSticks * sizeof(Stick),
		cudaMemcpyHostToDevice);
	cudaMalloc(&device_waters, nbWaters * sizeof(Water));
	cudaMemcpy(
		device_waters,
		host_waters,
		nbWaters * sizeof(Water),
		cudaMemcpyHostToDevice);
	cudaMalloc(&device_foods, nbFoods * sizeof(Food));
	cudaMemcpy(
		device_foods,
		host_foods,
		nbFoods * sizeof(Food),
		cudaMemcpyHostToDevice);
	cudaMalloc(&device_anthills, nbAnthills * sizeof(Anthill));
	cudaMemcpy(
		device_anthills,
		host_anthills,
		nbAnthills * sizeof(Anthill),
		cudaMemcpyHostToDevice);
	cudaMalloc(&device_ants, nbAnts * sizeof(Ant));
	cudaMemcpy(
		device_ants,
		host_ants,
		nbAnts * sizeof(Ant),
		cudaMemcpyHostToDevice);
	// Libere la memoire cote CPU
	free(host_rocks);	
	free(host_sticks);
	free(host_waters);
	free(host_foods);
	free(host_anthills);
	free(host_ants);

	// Dessine la carte, les ressources, les pheromones, les fourmis pour la premiere fois
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_init_draw_map << <blocks, threads >> > (surfaceMap, 1024, 1024, device_rocks, nbRocks, device_sticks, nbSticks);
	kernel_init_draw_ressources << <blocks, threads >> > (surfaceRessources, surfaceMap, 1024, 1024, device_waters, nbWaters, device_foods, nbFoods, device_anthills, nbAnthills);
	kernel_init_draw_pheromone << <blocks, threads >> > (surfacePheromones, surfaceRessources, 1024, 1024);
	kernel_init_draw_ants << <blocks, threads >> > (surfaceAnts, 1024, 1024);
}

// Libere toute la memoire allouee sur le GPU
void destroyWorld() {
	if (device_rocks)
		cudaFree(device_rocks);
	if (device_sticks)
		cudaFree(device_sticks);
	if (device_waters)
		cudaFree(device_waters);
	if (device_foods)
		cudaFree(device_foods);
	if (device_anthills)
		cudaFree(device_anthills);
	if (device_ants)
		cudaFree(device_ants);
}

// Genere l'image pour chaque frame et gere la simulation
void GenereImage(cudaSurfaceObject_t surfaceOut, cudaSurfaceObject_t surfaceMap, cudaSurfaceObject_t surfaceRessources, cudaSurfaceObject_t surfacePheromones, cudaSurfaceObject_t surfaceAnts, char * str) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	// On fusionne Map et Ressource
	kernel_fuse_map << <blocks, threads >> > (surfaceOut, surfaceMap, surfaceRessources, 1024, 1024);
	// On bouge les pheromones
	kernel_disperse_pheromones << <blocks, threads >> > (surfacePheromones, 1024, 1024);
	kernel_dillution_pheromones << <blocks, threads >> > (surfacePheromones, 1024, 1024);
	threads = dim3(nbFoods);
	blocks = dim3(1);
	// On regenere toutes les ressources qui ont besoin d'etre regenerees 
	kernel_regen_ressources << <blocks, threads >> > (surfaceRessources, surfaceMap, 1024, 1024, device_waters,nbWaters, device_foods, nbFoods, device_rocks, nbRocks);
	
	dim3 threads2(1024);
	// Marche tant que nbAnts%1024 == 0
	dim3 blocks2(nbAnts / 1024);
	// On gere l'affichage et le comportement des fourmis
	kernel_draw_ant << <blocks2, threads2 >> > (surfaceOut, surfaceRessources, surfacePheromones, surfaceAnts, 1024, 1024, device_anthills, nbAnthills, device_ants, nbAnts);
	// On recupere la anthill temporairement pour afficher ses statistiques d'eau et nourriture 
	Anthill * tmp_host_anthill = (Anthill*)malloc(1*sizeof(Anthill));
	cudaMemcpy(
		tmp_host_anthill,
		device_anthills,
		sizeof(Anthill),
		cudaMemcpyDeviceToHost);
	sprintf(str,"Food : %d, Water : %d", tmp_host_anthill[0].food, tmp_host_anthill[0].water);
}

// Toutes les textures des differents elements
texture<float4, 2, cudaReadModeElementType> texBackground;
texture<float4, 2, cudaReadModeElementType> texRock;
texture<float4, 2, cudaReadModeElementType> texStick;
texture<float4, 2, cudaReadModeElementType> texWater;
texture<float4, 2, cudaReadModeElementType> texFood;
texture<float4, 2, cudaReadModeElementType> texAnthill;

// Applique une texture de tous les elements (sauf fourmis vu qu'elles font 1 pixel)
__global__ void applyTexture(cudaSurfaceObject_t surface, uint32_t width, uint32_t height) {
	const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 colorpix;
	surf2Dread(&colorpix, surface, x * sizeof(float4), y);
	// nombre de fois que la texture se repete su l'ecran de 1024(<=64 pour etre coherent avec la texture de base 16x16) 
	const int n = 16;
	// Calcule la coordonnee dans la texture
	const float sizeTexX = (float)width / (float)n;
	const float sizeTexY = (float)height / (float)n;
	const float tx = 16.f * modf(x, sizeTexX) / sizeTexX;
	const float ty = 16.f * modf(y, sizeTexY) / sizeTexY;
	if (colorpix == COLOR_BACKGROUND) colorpix = tex2D(texBackground, tx, ty);
	else if (colorpix == COLOR_ROCK) colorpix = tex2D(texRock, tx, ty);
	else if (colorpix == COLOR_STICK) colorpix = tex2D(texStick, tx, ty);
	else if (colorpix == COLOR_WATER) colorpix = tex2D(texWater, tx, ty);
	else if (colorpix == COLOR_FOOD) colorpix = tex2D(texFood, tx, ty);
	else if (colorpix == COLOR_ANTHILL) colorpix = tex2D(texAnthill, tx, ty);
	else return;
	surf2Dwrite(colorpix, surface, x * sizeof(float4), y);
}

// Copie une SurfaceIn dans une surfaceOut
void showOnly(cudaSurfaceObject_t surfaceOut, cudaSurfaceObject_t surfaceIn) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_copy_map << <blocks, threads >> > (surfaceOut, surfaceIn, 1024, 1024);
}

// Applique la texture de tous les elements
void applyTex2D(cudaSurfaceObject_t surface, char linear) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	texBackground.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
	texRock.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
	texStick.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
	texWater.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
	texFood.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;
	texAnthill.filterMode = linear ? cudaFilterModeLinear : cudaFilterModePoint;

	applyTexture << <blocks, threads >> > (surface, 1024, 1024);
}

// Recupere les valeurs des pixels d'une texture 16x16 a partir d'un pointeur de fichier
char loadBMP(float4 OutArr[16][16],FILE * fptr){
	if (!fptr) {
		printf("probleme d'ouverture");
		return 0;
	}
	// Parseur BMP
	unsigned char line[768] = {0};
	uint16_t idf,ida,ida2;
	uint32_t size,offset;
	fread(&idf,2,1,fptr);
	fread(&size,4,1,fptr);
	fread(&ida,2,1,fptr);
	fread(&ida2,2,1,fptr);
	fread(&offset, 4, 1, fptr);
	int taille = size - offset;
	if (taille != 768) {
		printf("probleme de taille");
		return 0;
	}
	int sumread = 0;
	int read = 1;
	fseek(fptr, offset, SEEK_SET);
	while(sumread<768 && !ferror(fptr) && read){
		read = fread(&line[sumread], 1, 768-sumread, fptr);
		sumread += read;
		if(feof(fptr)){// Gere le cas ou la structure FILE croit etre en fin de fichier mais devrait lire un 0 
			line[sumread] = 0;
			sumread++;
			fseek(fptr, offset + sumread, SEEK_SET);
		}
	}
	if(sumread<768){
		printf("size %u, offset %u\n",size,offset);
		printf("probleme de lecture %d/%d last read = %d ",sumread,768,line[read-1]);
		printf("%s %s\n",feof(fptr)?"Fin de fichier":"",ferror(fptr)?"Erreur":"");
		return 0;
	}
	for(int i = 0;i<taille;i+=3){
		int index = i/3;
		OutArr[index/16][index%16] = make_float4((float)line[i+2]/255.f,(float)line[i+1]/255.f,(float)line[i]/255.f,1.f);
	}
	return 1;
}

// Transforme un tableau de 16x16 en texture 2D de 16x16
char fromArrayToTex(float4 Array[16][16],texture<float4, 2, cudaReadModeElementType> *tex){
	size_t pitch, tex_ofs;
	float4 *ArrayDevice = 0;
	// Creation d'un tableau device pour accueillir le tableau 2d host
	cudaMallocPitch((void**)&ArrayDevice,&pitch,16*sizeof(float4),16);
	// Copie du contenu de Array dans ArrayDevice
	cudaMemcpy2D(ArrayDevice, pitch, Array, 16*sizeof(float4),16*sizeof(float4),16,cudaMemcpyHostToDevice);
	tex->normalized = false;

	// Association de la texture a son tableau de valeur
	cudaBindTexture2D (&tex_ofs, tex, ArrayDevice, &tex->channelDesc,16, 16, pitch);
	//tex->filterMode = cudaFilterModeLinear;
	// On verifie l'absence d'offset
	if (tex_ofs) {
		return 0;
	}
	return 1;
}

// Charge en memoire toutes les textures de tous les elements 
char loadBMPs(FILE* fptrBackground, FILE* fptrRock, FILE* fptrStick, FILE* fptrWater, FILE* fptrFood, FILE* fptrAnthill) {
	float4 hostArray[16][16];
	printf("Load Background\n");
	if(!loadBMP(hostArray,fptrBackground)) return 0;
	if(!fromArrayToTex(hostArray,&texBackground)) return 0;	
	printf("End Background\n");
	printf("Load Rock\n");
	if(!loadBMP(hostArray,fptrRock)) return 0;
	if(!fromArrayToTex(hostArray,&texRock)) return 0;	
	printf("End Rock\n");
	printf("Load Stick\n");
	if(!loadBMP(hostArray,fptrStick)) return 0;
	if(!fromArrayToTex(hostArray,&texStick)) return 0;	
	printf("End Stick\n");
	printf("Load Water\n");
	if(!loadBMP(hostArray,fptrWater)) return 0;
	if(!fromArrayToTex(hostArray,&texWater)) return 0;	
	printf("End Water\n");
	printf("Load Food\n");
	if(!loadBMP(hostArray,fptrFood)) return 0;
	if(!fromArrayToTex(hostArray,&texFood)) return 0;	
	printf("End Food\n");
	printf("Load Anthill\n");
	if(!loadBMP(hostArray,fptrAnthill)) return 0;
	if(!fromArrayToTex(hostArray, &texAnthill)) return 0;
	printf("End Anthill\n");
	return 1;
}
