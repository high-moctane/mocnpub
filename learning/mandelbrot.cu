// マンデルブロ集合の GPU カーネル

// デバイス関数：マンデルブロ計算（発散するまでの反復回数を返す）
__device__ int mandelbrot_calc(float cx, float cy, int max_iter) {
    float zx = 0.0f;
    float zy = 0.0f;

    for (int n = 0; n < max_iter; n++) {
        // z の絶対値の2乗を計算（|z|^2 = zx^2 + zy^2）
        float zx2 = zx * zx;
        float zy2 = zy * zy;

        // |z| >= 2.0 なら発散
        if (zx2 + zy2 >= 4.0f) {
            return n;
        }

        // z = z^2 + c を計算
        // (zx + i*zy)^2 = (zx^2 - zy^2) + i*(2*zx*zy)
        float new_zx = zx2 - zy2 + cx;
        float new_zy = 2.0f * zx * zy + cy;

        zx = new_zx;
        zy = new_zy;
    }

    return max_iter;  // 発散しなかった
}

// カーネル：マンデルブロ集合を計算
extern "C" __global__ void mandelbrot_kernel(
    unsigned int *result,  // 結果を書き込む配列（反復回数）
    int width,             // 画像の幅
    int height,            // 画像の高さ
    float x_min,           // 複素数平面の x 最小値
    float x_max,           // 複素数平面の x 最大値
    float y_min,           // 複素数平面の y 最小値
    float y_max,           // 複素数平面の y 最大値
    int max_iter           // 最大反復回数
) {
    // 自分のスレッドIDを計算（担当するピクセル座標）
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // 画像サイズを超えるスレッドは何もしない
    if (px >= width || py >= height) {
        return;
    }

    // ピクセル座標 → 複素数平面の座標に変換
    float cx = x_min + ((float)px / (float)width) * (x_max - x_min);
    float cy = y_min + ((float)py / (float)height) * (y_max - y_min);

    // マンデルブロ計算
    int iter = mandelbrot_calc(cx, cy, max_iter);

    // 結果を書き込む
    result[py * width + px] = iter;
}
