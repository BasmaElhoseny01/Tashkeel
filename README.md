# Tashkeel

<img src="https://wefaqdev.net/upload/7747943442.jpg" height="500px"/>



## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents
- <a href ="#Overview">Overview</a>
- <a href ="#Achievement">Our Achievement</a>
- <a href ="#started"> Get Started</a>
- <a href ="#modules"> Modules</a>
  - <a href ="#preprocessing">Preprocessing</a>
  - <a href ="#network">Network</a>
- <a href ="#contributors">Contributors</a>
- <a href ="#license">License</a>


## <img align="center"  width =50px  height =40px src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/waving-hand_1f44b.gif"> Overview <a id = "Overview"></a>
-------------------------------- Given an image containing a single hand 🖐, our system classifies the hand gesture into one of six digits <em>from 0️⃣ to 5️⃣</em>. It handles different lighting effects⚡ and different hand poses as much as possible.



## <img align="center"  width =60px  height =70px src="https://opengameart.org/sites/default/files/gif_3.gif"> Our Achievement <a id = "Achievement"></a>
 ....................................................  We  have been ranked as the **3rd team** on the leader-board out of 20 teams with an accuracy of **73%** on the hidden test set 
 ![image](https://github.com/BasmaElhoseny01/Tashkeel/assets/72309546/dc9bb03e-7adb-48e3-9214-6918b3214cb2)



## <img  align= center width=50px height=50px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif">Get Started <a id = "started"></a>
...........................................................

<a id = "modules"></a>
##  <img align="center"  width =60px  height =70px src="https://media0.giphy.com/media/QcvMA5Oebq1sk/giphy.gif?cid=6c09b9524f0j11h02l3m2i5ghx4et60zmj4vxls7d3z6xzr8&ep=v1_gifs_search&rid=giphy.gif&ct=g"> Modules

<a id = "preprocessing"></a>
### <img align="center"  width =60px  height =70px src="https://media4.giphy.com/media/ux6vPam8BubuCxbW20/giphy.gif?cid=6c09b952gi267xsujaqufpqwuzeqhbi88q0ohj83jwv6dpls&ep=v1_stickers_related&rid=giphy.gif&ct=s"> Preprocessing
  #####  <img align="center"  width =50px  src="https://maidcleantx.com/wp-content/uploads/2017/11/broom-gif.gif"> Cleaning Process  [Train & Validation Only]
      1. Remove HTML tags
      2. Remove URLs
      3. Remove special Arabic character (Kashida)
      4. Separate Numbers
      5. Remove Multiple Whitespaces
      6. Clear Punctuations
      7. Remove english letters and english and arabic numbers
      8. Remove shifts
  #####  <img align="center"  width =40px  src="https://i2.wp.com/media0.giphy.com/media/4KELPefVuGnAvPJ2lx/giphy.gif"> Tokenization 
      • Split Using: [\n.,،؛:«»?؟]+
  #####  <img align="center"  width =40px  src="https://media2.giphy.com/media/2W9HV0KOywNhnkKl6O/giphy.gif"> Fix Diacritization Issue [Train & Validation Only]
      1. Replace consecutive diacritics with a single diacritic
      2. Ending Diacritics: Remove diacritics at the end of a word
      3. Misplaced Diacritics: Remove spaces between characters and diacritics
  #####  <img align="center"  width =40px  src="https://ugokawaii.com/wp-content/uploads/2023/06/trash-can.gif"> Tashkel Removal [Train & Validation Only]
      • Remove gold class for every character
      • Harakat:
        1. "Fatha":"\u064e"
        2. "Fathatan":  "\u064b"
        3. "Damma":"\u064f"
        4. "Dammatan":"\u064c"
        5. "Kasra":"\u0650"
        6. "Kasratan":"\u064d"
        7. "Sukun":"\u0652"
        8. "Shadda":"\u0651"
        9. "Shadda Fatha":"\u0651\u064e"
        10. "Shadda Fathatan":"\u0651\u064b"
        11. "Shadda Damma":"\u0651\u064f"
        12. "Shadda Dammatan":"\u0651\u064c"
        13. "Shadda Kasra":"\u0651\u0650"
        14. "Shadda Kasratan":"\u0651\u064d"      
    

##### Reference <a href="https://arxiv.org/abs/1905.01965">Arabic Text Diacritization Using Deep Neural Networks</a> 

<a id = "network"></a>
### <img align="center"  width =60px  height =70px src="https://static.wixstatic.com/media/17ac83_cf1a5fed37844786aafa17eca78679eb~mv2.gif"> Network
class Tashkeel(nn.Module):
  def __init__(self, vocab_size=vocab_size, embedding_dim=100, hidden_size=256, n_classes=n_classes):
    """
    The constructor of our Tashkeel model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)
    """
    super(Tashkeel, self).__init__()
    # (1) Create the embedding layer
    self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)

    # (2) Create an LSTM layer with hidden size = hidden_size and batch_first = True
    # self.lstm =  nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)
    self.lstm =  nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True,num_layers=2,bidirectional=True)

    # (3) Create a linear layer with number of neorons = n_classes
    self.linear =  nn.Linear(2*hidden_size,n_classes)
##### Reference <a href="https://ieeexplore.ieee.org/document/9274427">Effective Deep Learning Models for Automatic Diacritization of Arabic Text</a> 
  



<!-- Contributors -->
## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif?cid=6c09b952tmewuarqtlyfot8t8i0kh6ov6vrypnwdrihlsshb&rid=giphy.gif&ct=s"> Contributors <a id = "contributors"></a>

<!-- Contributors list -->
<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/Ahmed-H300"><img src="https://avatars.githubusercontent.com/u/67925988?v=4" width="150px;" alt=""/><br /><sub><b>Ahmed Hany</b></sub></a></td>
    <td align="center"><a href="https://github.com/Mohabz-911" ><img src="https://avatars.githubusercontent.com/u/68201932?v=4" width="150px;" alt=""/><br /><sub><b>Mohab Zaghloul</b></sub></a><br />
    <td align="center"><a href="https://github.com/ShazaMohamed"><img src="https://avatars.githubusercontent.com/u/56974730?v=4" width="150px;" alt=""/><br /><sub><b>Shaza Mohamed</b></sub></a><br />
    <td align="center"><a href="https://github.com/BasmaElhoseny01"><img src="https://avatars.githubusercontent.com/u/72309546?v=4" width="150px;" alt=""/><br /><sub><b>Basma Elhoseny</b></sub></a><br /></td>
  </tr>
</table>

## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/ggoKD4cFbqd4nyugH2/giphy.gif?cid=6c09b9527jpi8kfxsj6eswuvb7ay2p0rgv57b7wg0jkihhhv&rid=giphy.gif&ct=s"> License <a id = "license"></a>
This software is licensed under MIT License, See [License](https://github.com/BasmaElhoseny01/Tashkeel/blob/main/LICENSE) for more information ©Basma Elhoseny.
