# Cat Interactive AR
Cat Interactive AR โปรเจกต์ที่เน้นการจำลองประสบการณ์โต้ตอบกับแมวเสมือนนโลกจริง โดยผสานเทคโนโลยี Augmented Reality (AR) เข้ากับ Computer Vision AI เพื่อให้ผู้ใช้สามารถ ใช้มือในการสื่อสาร เล่น หรือโต้ตอบกับแมวเสมือน ได้อย่างเป็นธรรมชาติและเรียลไทม์

ระบบหลักทำงานด้วย AR Foundation และ Mediapipe Hand Landmarker สำหรับตรวจจับ Landmark ของมือ 21 จุดจากนั้นวิเคราะห์ท่าทางและ mapping เข้าสู่โลกสามมิติ เพื่อให้แมวตอบสนองต่อผู้เล่นอย่างสมจริงโดยมี HandARController.cs เป็นตัวควบคุมหลักในการจัดการ ข้อมูล การประมวลผล และการโต้ตอบทั้งหมด
#### โครงสร้างโปรเจกต์
```
CatInteractiveAR/
├── Assets/
├── Final_Build_And_Markers/  #ตัวโปรเจคต์ที่ built เสร้จแล้ว ทดลองได้
├── Packages/         
├── Projectsetting/       
└── README.md
```

#### วิธีติดตั้ง
1. Clone โปรเจกต์
``` git clone https://github.com/sSODAs/Cat_Interactive_AR.git ```

2. เปิด Unity Hub แล้วเพิ่มโปรเจกต์

3. เปิด Scene หลักและรันใน Editor หรือ Build ไปยังอุปกรณ์ AR
