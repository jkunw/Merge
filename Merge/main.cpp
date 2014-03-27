#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* Patterns of the barcode in four glyph markers.
`1` stands for white cell and `0` for black cell.
For example, one glyph pattern will be represented as
        b b b b b
        b b w b b      0 1 0
        b w w w b  =>  1 1 1  =>  0 1 0 1 1 1 1 0 1.
        b w b w b      1 0 1
        b b b b b
The pattern can be seen from four directions, so pattern list has four rows.
*/
int pattern1[] = {0, 1, 0, 1, 1, 1, 1, 0, 1,
                  1, 1, 0, 0, 1, 1, 1, 1, 0,
                  1, 0, 1, 1, 1, 1, 0, 1, 0,
                  0, 1, 1, 1, 1, 0, 0, 1, 1};
int pattern2[] = {0, 0, 1, 1, 0, 1, 0, 1, 0,
                  0, 1, 0, 1, 0, 0, 0, 1, 1,
                  0, 1, 0, 1, 0, 1, 1, 0, 0,
                  1, 1, 0, 0, 0, 1, 0, 1, 0};
int pattern3[] = {0, 0, 1, 0, 1, 0, 1, 0, 1,
                  1, 0, 0, 0, 1, 0, 1, 0, 1,
                  1, 0, 1, 0, 1, 0, 1, 0, 0,
                  1, 0, 1, 0, 1, 0, 0, 0, 1};
int pattern4[] = {0, 1, 0, 0, 1, 1, 0, 0, 1,
                  0, 0, 0, 0, 1, 1, 1, 1, 0,
                  1, 0, 0, 1, 1, 0, 0, 1, 0,
                  0, 1, 1, 1, 1, 0, 0, 0, 0};
const int *pattern[] = {pattern1, pattern2, pattern3, pattern4};

const int APPROX_POLY_EPSILON = 3;  // Epsilon for `approxPolyDP`.
const int MIN_CONTOUR_AREA    = 50; // Minimal contour area to filter contours.
const int CELL_NUM_ROW        = 3;  // Cell number per row.
const int GLYPH_SIZE          = 30; // Resized glyph size.
const int CELL_NUM            = CELL_NUM_ROW * CELL_NUM_ROW; // Total cell number.
const int CELL_SIZE           = GLYPH_SIZE / CELL_NUM_ROW;   // Resized cell size.

const int MAX_DELAY_FRAME     = 3;  // Maximal delay frame.
int delay                     = MAX_DELAY_FRAME; // Current delay frame.
Mat last_m; // Transformation matrix last time.

// Dst points to normalize glyph area to square.
// Clockwise 
vector<Point2f> cw_dst_points  = {Point2f(-CELL_SIZE, -CELL_SIZE),
                                  Point2f(GLYPH_SIZE + CELL_SIZE, -CELL_SIZE),
							      Point2f(GLYPH_SIZE + CELL_SIZE, GLYPH_SIZE + CELL_SIZE),
							      Point2f(-CELL_SIZE, GLYPH_SIZE + CELL_SIZE)};
// Counterclockwise
vector<Point2f> ccw_dst_points = {Point2f(-CELL_SIZE, -CELL_SIZE),
                                  Point2f(-CELL_SIZE, GLYPH_SIZE + CELL_SIZE),
								  Point2f(GLYPH_SIZE + CELL_SIZE, GLYPH_SIZE + CELL_SIZE),
								  Point2f(GLYPH_SIZE + CELL_SIZE, -CELL_SIZE)};

int width  = 640; // Width of image.
int height = 480; // Height of image.


/* Find glyph centers in image `src`.
Return true if all four glyphs are detected.
*/
bool find_glyphs(const Mat &src, vector<Point2f> &glyph_centers) {
    // Convert to grayscale.
    Mat gs_src;
    if(src.type() != CV_8UC1)
        cvtColor(src, gs_src, COLOR_BGR2GRAY);
    else
        gs_src = src;

    // Canny edge detection
    // Use OTSH threshold as Canny high threshold.
    Mat bw_src;
    Mat edges;
    double high_threshold = threshold(gs_src, bw_src, 0, 255, THRESH_BINARY | THRESH_OTSU);
    Canny(gs_src, edges, 0.5*high_threshold, high_threshold);

    // Find contours from edges.
    // RETR_LIST: retrieve all contours;
    // CHAIN_APPROX_SIMPLE: compress contours;
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Approximate contours with polygen 
    // and select convex quadrilaterals with area greater than MIN_CONTOUR_AREA.
    vector<vector<Point>> vertices;
    for(const auto &cnt : contours) {
        vector<Point> v;
        approxPolyDP(cnt, v, APPROX_POLY_EPSILON, true);
        if(v.size() == 4 &&
           contourArea(v) > MIN_CONTOUR_AREA &&
           isContourConvex(v))
           vertices.push_back(v);
    }
    if(vertices.size() < 4) // The image contains less than 4 glyph markers.
        return false;

    // For every quadrilateral, check if it matches a pattern.
    int found[4] = {0, 0, 0, 0}; // Four `found` flag.
    for(const auto &v : vertices) {
        // Convert to float points.
        vector<Point2f> fv(4);
        for(int i = 0; i < 4; i++) {
            Point2f f = Point(v[i]);
            fv[i] = f;
        }
        
        // Get glyph transformation matrix.
        Mat m;
        // If contour area < 0, four vertices of the contour is stored clockwise.
        if(contourArea(v, true) < 0)
            m = getPerspectiveTransform(fv, cw_dst_points);
        // Otherwise, they are stored counterclockwise.
        else
            m = getPerspectiveTransform(fv, ccw_dst_points);

        // Transform quadrilateral to square.
        Size dsize(GLYPH_SIZE, GLYPH_SIZE);
        Mat glyph(dsize, CV_8UC1);
        warpPerspective(gs_src, glyph, m, dsize);

        // Convert to binary image.
        threshold(glyph, glyph, 0, 255, THRESH_BINARY | THRESH_OTSU);

        // Mesh the square to CELL_NUM_ROW X CELL_NUM_ROW grid.
        // Assume the cell is white if white pixels in the cell outnumbers black ones.
        // Store the color pattern in `p`.
        int p[CELL_NUM];
        for(int i = 0; i < CELL_NUM; i++) {
            int sum = 0;
            int row = (i / CELL_NUM_ROW) * CELL_SIZE;
            int col = (i % CELL_NUM_ROW) * CELL_SIZE;
            // Sum cell pixels.
            for(int r = row; r < row + CELL_SIZE; r++)
                for(int c = col; c<col + CELL_SIZE; c++)
                    sum += (int)glyph.at<uchar>(c, r);

            if(sum / 255 > CELL_SIZE*CELL_SIZE / 2)
                p[i] = 1; // White cell
            else
                p[i] = 0; // Black cell
        }

        // Compare current cell pattern to 4 patterns in 4 directions.
        // If matched pattern is found the first time, set the corresponding `found` flag.
        for(int j = 4, i = 0; i < 4 && j == 4; i++) {
            for(j = 0; j < 4; j++) {
                int k;
                for(k = 0; k < CELL_NUM; k++) {
                    if(pattern[i][j*CELL_NUM + k] != p[k])
                        break;
                }
                // Find a match pattern.
                if(k == CELL_NUM) {  
                    // The center of marker is the average of four vertices.
                    Point2f center = (v[0] + v[1] + v[2] + v[3]) * 0.25;
                    if(!found[i]) {
                        glyph_centers[i] = center;
                        found[i]++;
                        break;
                    }
                }
            }
        }
    }

    // If all four markers are found, return true.
    if(found[0] && found[1] && found[2] && found[3])
        return true;
    else
        return false;
}

/* Fuse src with dst.
*/
void fuse(Mat &src, Mat &dst, const vector<Point2f> &src_points) {
    // Convert src to grayscale.
    if (src.type() != CV_8UC1)
        cvtColor(src, src, COLOR_BGR2GRAY);

    /* Get transformation matrix according to markers in template and dst image.
    In order to avoid blinking when in some frames markers are not all detected,
    transformation matrix in the last time `last_m` is used. This may result in 
    delay when camera moves fast. The maximal times to use `last_m` is MAX_DELAY_FRAME.
    */
    Mat m;
    vector<Point2f> dst_points(4);
    if (find_glyphs(dst, dst_points)) {
        m = getPerspectiveTransform(src_points, dst_points);
        delay = 0;
        last_m = Mat(m);
    }
    else if (delay < MAX_DELAY_FRAME) {
        m = Mat(last_m);
        delay++;
    }
    else 
        return;

    // Transform src.
    warpPerspective(src, src, m, dst.size());

    // Merge transformed src and dst. 
    // White region in src will become green in the merged image.
    Mat bgr[3];
    split(dst, bgr);
    bgr[0] -= src;
    bgr[1] += src;
    bgr[2] -= src;
    merge(bgr, 3, dst);
}

int main()
{
    VideoCapture cam(0);
    if (!cam.isOpened())
        return -1;

    Mat templ = imread("template.jpg");
    vector<Point2f> src_points(4);
    if (!find_glyphs(templ, src_points))
        return -1;

    Mat dst, src;
    namedWindow("merge test", WINDOW_AUTOSIZE);
    while (true) {
        src = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        cam.read(dst);        
        fuse(src, dst, src_points);
        imshow("merge test", dst);
        if (waitKey(10) == 27)
            break;
    }
    destroyAllWindows();
    return 0;
}