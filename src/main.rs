use cudarc::driver::CudaContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 mocnpub - CUDA デバイステスト 🔥\n");

    // GPU 0 のコンテキストを作成
    println!("GPU デバイス 0 に接続中...");
    let ctx = CudaContext::new(0)?;

    println!("✅ GPU デバイス 0 に接続成功！");
    println!("🎉 CUDA が正常に動作しています！");

    // デフォルトストリームを取得（動作確認）
    let _stream = ctx.default_stream();
    println!("✅ デフォルトストリームを取得しました");

    Ok(())
}
