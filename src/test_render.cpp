#include "camera.cuh"
#include "test_render.h"
#include <cstdlib>
#include <vector>
#include <cfloat>
#include <X11/Xlib.h>
#include <GL/glut.h>

using namespace std;
using namespace geom;

// Screen quad
GLfloat vertices[] = {
	// positions			// UV
	-1.0f, 1.0f, 0.0f,		0.0f, 1.0f,
	1.0f, 1.0f, 0.0f,		1.0f, 1.0f,
	-1.0f, -1.0f, 0.0f,		0.0f, 0.0f,
	-1.0f, -1.0f, 0.0f,		0.0f, 0.0f,
	1.0f, 1.0f, 0.0f,		1.0f, 1.0f,
	1.0f, -1.0f, 0.0f,		1.0f, 0.0f
};

typedef vector<Triangle> TriVec;

struct BBox {
	int boxId;
	Vector3Df _bottom;
	Vector3Df _top;
	Vector3Df _color;
	int leftId, rightId;
	bool isLeaf;
	TriVec tris;
};

static int boxId;
vector<BBox> bboxes;

std::ostream& operator << (std::ostream& o, const Vector3Df &v) {
	o << "x: " << v.x << "\ty: " << v.y <<  "\tz: " << v.z << std::endl;
	return o;
}

std::ostream& operator << (std::ostream& o, const geom::Triangle *v) {
	o << "Triangle with ID: " << v->_triId << std::endl;
	return o;
}

std::ostream& operator << (std::ostream& o, const BBox &b) {
	o << "BBox ID: " << b.boxId;
	o << "Bottom:\t" << b._bottom;
	o << "Top: \t" << b._top;
	o << "Color: \t"<<b._color;
	if (b.isLeaf) {
		o << "Is a leaf and has tris:" << endl;
		for (auto tri: b.tris) {
			o << "Triangle with ID: " << tri._triId << std::endl;
		}
	} else {
		o << "Is an inner node with children:" << endl;
		o << "Left: " << b.leftId << " and Right: " << b.rightId << endl;
	}
	return o;
}

int AddBoxes(BVHNode *root)
{
	BBox bbox;
	bbox.boxId = boxId++;
	bbox._bottom = root->_bottom;
	bbox._top = root->_top;
	bbox._color = Vector3Df((float)(rand() % 255)/255.0f,
							(float)(rand() % 255)/255.0f,
							(float)(rand() % 255)/255.0f);
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		bbox.isLeaf = false;
		bbox.leftId = AddBoxes(p->_left);
		bbox.rightId = AddBoxes(p->_right);
		bboxes.push_back(bbox);
	}
	else
	{
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(root);
		bbox.isLeaf = true;
		TriVec tris;
		for (auto tri: p->_triangles) {
			cout << "Adding tri " << tri->_triId << endl;
			Triangle triangle = *tri;
			tris.push_back(triangle);
		}
		bbox.tris = tris;
		bboxes.push_back(bbox);
		return bbox.boxId;
	}
	return -1;
}

int hitsBox(const Ray& ray, BBox* bbox) {


	Vector3Df minBound, maxBound;
	minBound = bbox->_bottom;
	maxBound = bbox->_top;
	float Tnear = -FLT_MAX;
	float Tfar = FLT_MAX;
    if (ray.dir.x == 0.f) {						    \
	if (ray.origin.x < minBound.x) return -1;					    \
	if (ray.origin.x > maxBound.x) return -1;					    \
	} else {											    \
	float T1 = (minBound.x - ray.origin.x)/ray.dir.x;			    \
	float T2 = (maxBound.x - ray.origin.x)/ray.dir.x;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return -1;									    \
	if (Tfar < 0.f)	return -1;									    \
	}

    if (ray.dir.y == 0.f) {						    \
	if (ray.origin.y < minBound.y) return -1;					    \
	if (ray.origin.y > maxBound.y) return -1;					    \
	} else {											    \
	float T1 = (minBound.y - ray.origin.y)/ray.dir.y;			    \
	float T2 = (maxBound.y - ray.origin.y)/ray.dir.y;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return -1;									    \
	if (Tfar < 0.f)	return -1;									    \
	}

    if (ray.dir.z == 0.f) {						    \
	if (ray.origin.z < minBound.z) return -1;					    \
	if (ray.origin.z > maxBound.z) return -1;					    \
	} else {											    \
	float T1 = (minBound.z - ray.origin.z)/ray.dir.z;			    \
	float T2 = (maxBound.z - ray.origin.z)/ray.dir.z;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return -1;									    \
	if (Tfar < 0.f)	return -1;									    \
	}

	return bbox->boxId;
}

void render(Triangle* tris) {
	Vector3Df* img = new Vector3Df[width*height];
	srand(0);
	AddBoxes(scene.getSceneBVHPtr());
	Camera* camera = scene.getCameraPtr();
	for (auto box: bboxes) {
		cout << box << endl;
	}
	Triangle* triangles = scene.getTriPtr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++){
			int idx = width*i + j;
			Ray ray = camera->computeTestCameraRay(j, i);
			for (auto bbox: bboxes) {
				int box = hitsBox(ray, &bbox);
				if ((bool)box) {
					img[idx] += bbox._color * 0.05f;
				}
			}
			for (int k = 0; k < scene.getNumTriangles(); k++) {
				float u, v;
				if (triangles[k].intersect(ray, u, v) < FLT_MAX) {
					img[idx] += triangles[k]._colorDiffuse * 0.5;
				}
			}
		}
	}
}

GLuint tex = 0;
void display(GLuint width, GLuint height)
{
    int i;
    float x, y;

    std::vector< unsigned char > buf;
    buf.reserve( width * height * 3 );
    for( size_t y = 0; y < height; ++y )
    {
        for( size_t x = 0; x < width; ++x )
        {
            // flip vertically (height-y) because the OpenGL texture origin is in the lower-left corner
            // flip horizontally (width-x) because...the original code did so
            size_t i = (height-y) * width + (width);
            buf.push_back( (unsigned char)( std::min(double(1), image[i].x) * 255.0 ) );
            buf.push_back( (unsigned char)( std::min(double(1), image[i].y) * 255.0 ) );
            buf.push_back( (unsigned char)( std::min(double(1), image[i].z) * 255.0 ) );
        }
    }

    /* clear all pixels */
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, tex );
    glTexSubImage2D
        (
        GL_TEXTURE_2D, 0,
        0, 0,
        width, height,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        &buf[0]
        );

    glBegin( GL_QUADS );
    glTexCoord2i( 0, 0 );
    glVertex2i( -1, -1 );
    glTexCoord2i( 1, 0 );
    glVertex2i(  1, -1 );
    glTexCoord2i( 1, 1 );
    glVertex2i(  1,  1 );
    glTexCoord2i( 0, 1 );
    glVertex2i( -1,  1 );
    glEnd();

    glutSwapBuffers();
}

Vector3Df* testRenderWrapper(Scene& scene, int width, int height, int samples, int numStreams, bool &useTexMemory, int argc, char** argv) {
	GLuint gWidth, gHeight;
	gWidth = width; gHeight = height;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(gWidth, gHeight);
	glutInitWindowPosition(10,10);
	glutCreateWindow(argv[0]);
	glutDisplayFunc(display);

	glGenTextures( 1, &tex );
	glBindTexture( GL_TEXTURE_2D, tex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glTexImage2D( GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL );

	glutMainLoop();


	Vector3Df* img = new Vector3Df[width*height];
	srand(0);
	AddBoxes(scene.getSceneBVHPtr());
	Camera* camera = scene.getCameraPtr();
	for (auto box: bboxes) {
		cout << box << endl;
	}
	Triangle* triangles = scene.getTriPtr();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++){
			int idx = width*i + j;
			Ray ray = camera->computeTestCameraRay(j, i);
			for (auto bbox: bboxes) {
				int box = hitsBox(ray, &bbox);
				if ((bool)box) {
					img[idx] += bbox._color * 0.05f;
				}
			}
			for (int k = 0; k < scene.getNumTriangles(); k++) {
				float u, v;
				if (triangles[k].intersect(ray, u, v) < FLT_MAX) {
					img[idx] += triangles[k]._colorDiffuse * 0.5;
				}
			}
		}
	}
	return img;
}

void testRender(BBox* bboxPtr, Camera* camPtr, Vector3Df* imgPtr, int width, int height) {

}
