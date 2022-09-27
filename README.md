# Smart-JPEG
*This is a smarter Jpeg algorithm implementation in python*<br>
__Compression steps:__
1. Fisrt I read the image (img3)
2. Then I divided the image into 16 x 16 blocks
3. I calculated the amount of sharpness for each block and broke 30% of the blocks with the highest sharpness into 8 x 8 blocks
4. I applied cosine transformation on each block
5. After that I applied appropriate multiplier factors to each 8 x 8 or 16 x 16 block
6. At last I took the reverse conversion and connected the blocks to get the compressed photo
