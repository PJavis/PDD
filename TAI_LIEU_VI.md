# Tài liệu tiếng Việt — Mô phỏng nhánh cây kẽm (Zinc Dendrite)

Tài liệu này giải thích **chương trình giải những phương trình gì, tại sao, và bằng cách nào**.
Dựa trên bài báo:

> Jing, Xing, Zhang và cộng sự, *Dynamics of zinc dendritic growth in aqueous zinc-based flow batteries:
> Insights from phase field–Lattice-Boltzmann simulations*, **Chemical Engineering Journal 503 (2025) 158318**.

---

## 1. Bài toán vật lý là gì?

Trong pin kẽm dùng dung môi nước, ion kẽm `Zn²⁺` trong dung dịch nhận điện tử ở điện cực và
bám lại thành kim loại rắn. Nếu quá trình này không đều, kim loại mọc thành các **nhánh cây
(dendrite)** — những gai nhọn hình cành cây. Dendrite mọc dài có thể đâm thủng màng ngăn và làm
**chập/hỏng pin**. Mục tiêu mô phỏng: dự đoán *khi nào* và *hình dạng thế nào* các nhánh này mọc,
để hiểu cách kìm hãm chúng.

Chương trình mô phỏng đồng thời **4 trường vật lý** trên một lưới 2 chiều `Nx × Ny`:

| Ký hiệu | Ý nghĩa | Đơn vị (không thứ nguyên) |
|---------|---------|---------------------------|
| `c`   | **Trường pha** (phase field): `c=1` là kim loại rắn, `c=0` là dung dịch, `0<c<1` là mặt phân giới | — |
| `cp`  | **Nồng độ ion** `Zn²⁺` chuẩn hoá `c⁺/c₀` | — |
| `phi` | **Điện thế** trong dung dịch | V |
| `u`   | **Vận tốc dòng chảy** của dung dịch (nếu bật dòng) | ô lưới / bước |

---

## 2. Bốn phương trình được giải — GIẢI GÌ, TẠI SAO, BẰNG CÁCH NÀO

### 2.1. Trường pha `c` — Phương trình Allen–Cahn dị hướng (Kobayashi)

**Giải gì.** Sự lớn lên của kim loại rắn theo thời gian:

```
τ ∂c/∂t = ∇·(A² ∇c)  +  [số hạng dị hướng]  +  c(1−c)(c − 0.5 + m)
```

- `A(θ) = W₀ (1 + δ·cos(ω(θ − θⱼ)))` là **hệ số dị hướng 6 cánh** — làm mặt phân giới thích mọc
  theo vài hướng ưu tiên, tạo ra hình gai/cành thay vì hình tròn.
- `m = m_max · tanh(k_dep · S / k_ref)` là **động lực mọc**, với `S` lấy từ công thức Butler–Volmer.

**Tại sao.** Trường pha (phase field) là cách chuẩn để mô phỏng mặt phân giới di động mà **không phải
theo dõi biên rõ ràng** — biên rắn/lỏng được biểu diễn "nhoè" qua vùng `0<c<1`. Bài báo gốc dùng
dạng bảo toàn Cahn–Hilliard; ở đây dùng dạng Allen–Cahn (Kobayashi) vì nó cho **cùng một vật lý định
tính** của dendrite nhưng đơn giản và nhanh hơn.

**Vì sao dùng `tanh` chứ không phải `arctan`.** Với `arctan`, khi `k_dep·S` lớn thì động lực bị **bão hoà**
— tăng gấp đôi `k_dep` gần như không đổi kết quả, khiến các thanh trượt "trông như không có tác dụng".
Dạng `tanh` với hệ số `k_ref` giữ thanh trượt luôn ở vùng nhạy, nên `k_dep` và `Ds` đều thực sự làm
đổi hình dạng.

**Bằng cách nào.** Sai phân hữu hạn (finite difference) hiện (explicit), toán tử Laplace/gradient/divergence
5 điểm, cập nhật `c += dt·(…)/τ` rồi kẹp `c` về `[0,1]`.

### 2.2. Nồng độ ion `Zn²⁺` — Phương trình Nernst–Planck

**Giải gì.** Ion kẽm di chuyển và bị tiêu thụ:

```
∂c⁺/∂t = ∇·(D_eff ∇c⁺)      (khuếch tán)
       + ∇·(D_eff c⁺ · zF/RT · ∇φ)   (điện di — bị điện trường kéo)
       − u·∇c⁺               (đối lưu theo dòng chảy, nếu bật)
       − [số hạng tiêu thụ]  (ion biến thành kim loại ở chỗ c tăng)
```

**Tại sao.** Nhánh cây mọc nhanh hay chậm phụ thuộc vào **có đủ ion tới nơi hay không**. Nernst–Planck
mô tả đủ 3 cơ chế vận chuyển ion: khuếch tán (theo chênh lệch nồng độ), điện di (theo điện trường), và
đối lưu (theo dòng chảy). Số hạng tiêu thụ nối trực tiếp với tốc độ mọc của `c` — mọc tới đâu ăn ion tới đó,
tạo ra **lớp nghèo ion** quanh đầu nhánh (hiệu ứng đầu nhọn — "tip effect", Hình 2 bài báo).

**Bằng cách nào.** Cũng bằng sai phân hiện. `D_eff = De·c + Ds·(1−c)` (khuếch tán chậm trong rắn, nhanh
trong lỏng). Gradient điện thế được **kẹp** (`grad_phi_cap`) để tránh phát tán số. Biên: nguồn ion liên
tục `=1` ở đỉnh và bên phải.

### 2.3. Điện thế `phi` — Phương trình Laplace (Poisson)

**Giải gì.** Phân bố điện thế trong dung dịch:

```
∇²φ = 0     (trong vùng dung dịch)
φ = φ_dep   (bị ghim tại vùng kim loại, c > 0.5)
```

**Tại sao.** Điện trường (`∇φ`) vừa kéo ion (điện di ở 2.2) vừa quyết định quá thế `η = φ − E_θ` trong
động lực Butler–Volmer (2.1). Vật dẫn kim loại là **đẳng thế**, nên vùng `c>0.5` bị ghim về một giá trị.
Đầu nhọn tập trung điện trường mạnh hơn → mọc nhanh hơn → càng nhọn: đây chính là cơ chế **tự khuếch đại**
làm hình thành dendrite.

**Bằng cách nào.** Lặp **Jacobi** (một dạng giải Laplace ổn định), khởi động ấm từ bước trước. Để nhanh,
chỉ **giải lại điện thế sau mỗi `phi_every` bước** (mặc định 4) vì `φ` biến đổi chậm so với `c`.

### 2.4. Dòng chảy `u` — Lattice–Boltzmann D2Q9 (tuỳ chọn)

**Giải gì.** Trường vận tốc dòng điện giải khi `u_inlet > 0`, gồm cả lực cản khuếch tán tại mặt phân giới
(Eq. 5–9 bài báo).

**Tại sao.** Trong pin dòng (flow battery) dung dịch chảy qua điện cực. Dòng chảy **đẩy ion về một phía**,
làm nhánh cây **nghiêng về phía dòng tới** (Hình 5–6 bài báo). Nếu để `u_inlet = 0` thì bỏ qua phần này.

**Bằng cách nào.** Phương pháp **Lattice–Boltzmann** D2Q9 mô hình BGK: va chạm (collision) + truyền
(streaming), bounce-back ở tường trên/dưới, biên vào cân bằng ở bên phải, chảy ra sao chép ở bên trái.
Vận tốc thu được được đưa ngược vào số hạng đối lưu của Nernst–Planck.

---

## 3. Cách ghép nối và trình tự tính (một bước thời gian)

Mỗi bước lặp `it` thực hiện theo thứ tự:

1. **Dòng chảy** (nếu bật): một bước Lattice–Boltzmann → cập nhật `u`.
2. **Điện thế**: cứ mỗi `phi_every` bước, giải lại Laplace cho `φ`.
3. **Động lực Butler–Volmer**: tính quá thế `η`, biểu thức `S`, động lực `m = m_max·tanh(k_dep·S/k_ref)`.
4. **Trường pha**: cập nhật `c` (khuếch tán dị hướng + phản ứng), kẹp về `[0,1]`, áp điều kiện biên.
5. **Ion**: cập nhật `c⁺` (khuếch tán + điện di + đối lưu − tiêu thụ), kẹp `[0,5]`, áp biên nguồn.
6. **Ghi hình**: cứ mỗi `record_every` bước, lưu ảnh và chiều cao đỉnh nhánh cao nhất.

---

## 4. Số Damköhler — "núm xoay" chính của hình dạng

Toàn bộ **hình thái** (compact hay phân nhánh) do một tỉ số điều khiển:

```
Da ≈ k_dep / Ds     (tốc độ phản ứng / tốc độ vận chuyển ion)
```

| Da | Chế độ | Hình dạng |
|----|--------|-----------|
| **cao** (Da > 1.5) | giới hạn bởi phản ứng | ion bị ăn hết trước khi tới đỉnh → **nhánh cây rậm, nhiều cành** |
| **thấp** (Da < 0.6) | giới hạn bởi vận chuyển | ion luôn dư → **khối đặc, tròn** |
| trung bình | hỗn hợp | vừa |

👉 Muốn thấy thay đổi rõ, hãy chỉnh `k_dep` và `Ds` theo **hai hướng ngược nhau**
(ví dụ `k_dep` cao + `Ds` thấp = rậm rạp).

---

## 5. Ý nghĩa các thanh trượt & khoảng giá trị

| Thanh trượt | Ý nghĩa | Khoảng |
|-------------|---------|--------|
| `k_dep`   | Tốc độ phản ứng (∝ mật độ dòng trao đổi i₀) | 0.5 – 80 |
| `Ds`      | Tốc độ khuếch tán/vận chuyển ion Zn²⁺ | 0.1 – 10 |
| `E_theta` | Điện thế cân bằng (càng âm càng "đẩy mạnh") | −0.8 … −0.02 V |
| `delta`   | Độ dị hướng (độ phân nhánh) | 0.0 – 0.6 |
| `u_inlet` | Độ mạnh dòng chảy dung dịch (0 = tĩnh, 2 = mạnh nhất) | 0.0 – 2.0 |
| `steps`   | Số bước thời gian (càng nhiều → nhánh càng cao, càng lâu) | 1000 – 24000 |
| **Quality / grid** | Kích thước lưới: chọn preset **hoặc** `Custom` để tự đặt | — |
| `Nx`      | **Chiều rộng** lưới (chỉ dùng khi chọn *Custom*) | 64 – 400 |
| `Ny`      | **Chiều cao** lưới (chỉ dùng khi chọn *Custom*) | 64 – 512 |
| Polycrystalline | Bật để mọc **nhiều mầm** cùng lúc (đa tinh thể) | — |
| `n_seeds` | Số mầm (khi bật đa tinh thể) | 2 – 24 |

> ⚠️ Lưới lớn (`Nx`, `Ny` cao) **cộng** với `steps` lớn thì chạy **chậm hơn nhiều**. Khi đang thử
> nghiệm nên để lưới nhỏ; chỉ tăng lên khi đã ưng hình dạng và muốn ảnh chi tiết cuối cùng.

Các khoảng trên đã được chọn nằm trong **giới hạn ổn định số** của sơ đồ sai phân hiện
(điều kiện `dt·Ds·4/dx² < 1`), nên không gây "nổ số" (giá trị vô cực/NaN).

> **`u_inlet` là "núm độ mạnh dòng chảy" 0–2.** Lattice–Boltzmann chỉ ổn định ở vận tốc thấp
> (Mach thấp, `< ~0.2` đơn vị lưới), nên giá trị thanh trượt được **ánh xạ nội bộ** về một vận tốc
> lưới an toàn (`2.0 → 0.2` đơn vị lưới). Nhờ vậy để số lớn vẫn cho dòng mạnh hơn mà **không bị NaN**.
> Muốn đổi trần này, sửa `U_INLET_MAX_LATTICE` trong `app.py`.

---

## 6. Đơn giản hoá so với bài báo gốc

- Trường pha dùng Allen–Cahn (Kobayashi) thay cho Cahn–Hilliard bảo toàn — vật lý định tính giống nhau.
- Các hằng số thứ nguyên được gộp vào các tốc độ không thứ nguyên có thể chỉnh (`k_dep`, `cs_c0`, …),
  với `W₀ = 1`, `τ₀ = W₀²/Ds = 1`.
- Lattice–Boltzmann: bounce-back tường trên/dưới, biên vào cân bằng bên phải, chảy ra sao chép bên trái;
  dòng chảy hoàn toàn do điều kiện biên vào điều khiển (không có lực khối).

---

## 7. Chạy chương trình

Giao diện web (khuyến nghị cho người mới):

```bash
uv run --with numpy --with matplotlib --with numba --with gradio python3 app.py
```

Mở đường dẫn hiện ra (mặc định `http://localhost:7860`). Dòng in `Backend: numba` nghĩa là đang chạy
engine nhanh. Nếu muốn hướng dẫn từng bước cho người chưa biết lập trình, xem **GETTING_STARTED.md**.
Chi tiết kỹ thuật tiếng Anh xem **README.md**.
