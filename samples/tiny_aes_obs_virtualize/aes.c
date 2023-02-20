#include "aes.h"

union _1_SubBytes_$node ;
enum _1_SubBytes_$op ;
static void SubBytes(void) ;
static void InvCipher(void) ;
static void AddRoundKey(uint8_t round ) ;
typedef uint8_t state_t[4][4];
enum _1_SubBytes_$op {
    _1_SubBytes__PlusPI_void_star_int2void_star$left_STA_0$result_STA_0$right_STA_1 = 123,
    _1_SubBytes__constant_void_star$result_STA_0$value_LIT_0 = 168,
    _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0 = 163,
    _1_SubBytes__local$result_STA_0$value_LIT_0 = 1,
    _1_SubBytes__PlusPI_void_star_unsigned_long2void_star$left_STA_0$result_STA_0$right_STA_1 = 190,
    _1_SubBytes__call$func_LIT_0 = 24,
    _1_SubBytes__constant_int$result_STA_0$value_LIT_0 = 58,
    _1_SubBytes__branchIfTrue$expr_STA_0$label_LAB_0 = 250,
    _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1 = 64,
    _1_SubBytes__PlusA_int_int2int$left_STA_0$result_STA_0$right_STA_1 = 103,
    _1_SubBytes__convert_int2unsigned_char$left_STA_0$result_STA_0 = 100,
    _1_SubBytes__global$result_STA_0$value_LIT_0 = 132,
    _1_SubBytes__convert_unsigned_char2int$left_STA_0$result_STA_0 = 3,
    _1_SubBytes__returnVoid$ = 90,
    _1_SubBytes__convert_unsigned_char2unsigned_long$left_STA_0$result_STA_0 = 136,
    _1_SubBytes__goto$label_LAB_0 = 91,
    _1_SubBytes__constant_unsigned_long$result_STA_0$value_LIT_0 = 35,
    _1_SubBytes__Lt_int_int2int$left_STA_0$result_STA_0$right_STA_1 = 182,
    _1_SubBytes__load_void_star$left_STA_0$result_STA_0 = 7,
    _1_SubBytes__Mult_unsigned_long_unsigned_long2unsigned_long$right_STA_0$result_STA_0$left_STA_1 = 154
} ;
unsigned char _1_SubBytes_$array[1][279]  = { {        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__constant_int$result_STA_0$value_LIT_0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__convert_int2unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1, 
            _1_SubBytes__goto$label_LAB_0,        (unsigned char)4,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__constant_int$result_STA_0$value_LIT_0,        (unsigned char)4,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0, 
            _1_SubBytes__convert_unsigned_char2int$left_STA_0$result_STA_0,        _1_SubBytes__Lt_int_int2int$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__branchIfTrue$expr_STA_0$label_LAB_0,        (unsigned char)14, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__goto$label_LAB_0, 
            (unsigned char)4,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__goto$label_LAB_0,        (unsigned char)232,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)1,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__constant_int$result_STA_0$value_LIT_0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__convert_int2unsigned_char$left_STA_0$result_STA_0, 
            _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1,        _1_SubBytes__goto$label_LAB_0,        (unsigned char)4,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__constant_int$result_STA_0$value_LIT_0,        (unsigned char)4, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0, 
            (unsigned char)1,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__convert_unsigned_char2int$left_STA_0$result_STA_0,        _1_SubBytes__Lt_int_int2int$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__branchIfTrue$expr_STA_0$label_LAB_0, 
            (unsigned char)14,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__goto$label_LAB_0,        (unsigned char)4,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__goto$label_LAB_0,        (unsigned char)157,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)2, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__constant_void_star$result_STA_0$value_LIT_0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__convert_unsigned_char2unsigned_long$left_STA_0$result_STA_0,        _1_SubBytes__constant_unsigned_long$result_STA_0$value_LIT_0, 
            (unsigned char)1,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__Mult_unsigned_long_unsigned_long2unsigned_long$right_STA_0$result_STA_0$left_STA_1,        _1_SubBytes__PlusPI_void_star_unsigned_long2void_star$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)1, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0, 
            _1_SubBytes__convert_unsigned_char2unsigned_long$left_STA_0$result_STA_0,        _1_SubBytes__constant_unsigned_long$result_STA_0$value_LIT_0,        (unsigned char)4,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__Mult_unsigned_long_unsigned_long2unsigned_long$right_STA_0$result_STA_0$left_STA_1,        _1_SubBytes__PlusPI_void_star_unsigned_long2void_star$left_STA_0$result_STA_0$right_STA_1, 
            _1_SubBytes__global$result_STA_0$value_LIT_0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__load_void_star$left_STA_0$result_STA_0,        _1_SubBytes__PlusPI_void_star_int2void_star$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0, 
            _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1,        _1_SubBytes__call$func_LIT_0,        (unsigned char)1,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__constant_void_star$result_STA_0$value_LIT_0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__convert_unsigned_char2unsigned_long$left_STA_0$result_STA_0,        _1_SubBytes__constant_unsigned_long$result_STA_0$value_LIT_0,        (unsigned char)1, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__Mult_unsigned_long_unsigned_long2unsigned_long$right_STA_0$result_STA_0$left_STA_1, 
            _1_SubBytes__PlusPI_void_star_unsigned_long2void_star$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)1,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__convert_unsigned_char2unsigned_long$left_STA_0$result_STA_0, 
            _1_SubBytes__constant_unsigned_long$result_STA_0$value_LIT_0,        (unsigned char)4,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__Mult_unsigned_long_unsigned_long2unsigned_long$right_STA_0$result_STA_0$left_STA_1,        _1_SubBytes__PlusPI_void_star_unsigned_long2void_star$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__global$result_STA_0$value_LIT_0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__load_void_star$left_STA_0$result_STA_0,        _1_SubBytes__PlusPI_void_star_int2void_star$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)3, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0, 
            _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)1,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__constant_int$result_STA_0$value_LIT_0,        (unsigned char)1, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0, 
            (unsigned char)1,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__convert_unsigned_char2int$left_STA_0$result_STA_0,        _1_SubBytes__PlusA_int_int2int$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__convert_int2unsigned_char$left_STA_0$result_STA_0, 
            _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1,        _1_SubBytes__goto$label_LAB_0,        (unsigned char)84,        (unsigned char)255, 
            (unsigned char)255,        (unsigned char)255,        _1_SubBytes__goto$label_LAB_0,        (unsigned char)79, 
            (unsigned char)255,        (unsigned char)255,        (unsigned char)255,        _1_SubBytes__local$result_STA_0$value_LIT_0, 
            (unsigned char)0,        (unsigned char)0,        (unsigned char)0,        (unsigned char)0, 
            _1_SubBytes__constant_int$result_STA_0$value_LIT_0,        (unsigned char)1,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        _1_SubBytes__local$result_STA_0$value_LIT_0,        (unsigned char)0,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__convert_unsigned_char2int$left_STA_0$result_STA_0, 
            _1_SubBytes__PlusA_int_int2int$left_STA_0$result_STA_0$right_STA_1,        _1_SubBytes__convert_int2unsigned_char$left_STA_0$result_STA_0,        _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1,        _1_SubBytes__goto$label_LAB_0, 
            (unsigned char)9,        (unsigned char)255,        (unsigned char)255,        (unsigned char)255, 
            _1_SubBytes__goto$label_LAB_0,        (unsigned char)4,        (unsigned char)255,        (unsigned char)255, 
            (unsigned char)255,        _1_SubBytes__goto$label_LAB_0,        (unsigned char)4,        (unsigned char)0, 
            (unsigned char)0,        (unsigned char)0,        _1_SubBytes__returnVoid$}};
static uint8_t xtime(uint8_t x ) ;
static void InvMixColumns(void) ;
static void MixColumns(void) ;
static void InvSubBytes(void) ;
static uint8_t getSBoxInvert(uint8_t num ) ;
static state_t *state ;
union _1_SubBytes_$node {
   char _char ;
   unsigned int _unsigned_int ;
   unsigned char _unsigned_char ;
   long _long ;
   unsigned long _unsigned_long ;
   void *_void_star ;
   unsigned short _unsigned_short ;
   unsigned long long _unsigned_long_long ;
   signed char _signed_char ;
   long long _long_long ;
   int _int ;
   short _short ;
};
static void BlockCopy(uint8_t *output , uint8_t const   *input ) ;
static void KeyExpansion(void) ;
static void InvShiftRows(void) ;
char const   *_1_SubBytes_$strings  =    "";
static uint8_t RoundKey[176] ;
static void ShiftRows(void) ;
static uint8_t input_save[16] ;
static uint8_t getSBoxValue(uint8_t num ) ;
static void Cipher(void) ;
void TestASM(void) ;
void megaInit(void) ;
static uint8_t *Key ;

AES_CONST_VAR uint8_t sbox[256] =   {
  //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

AES_CONST_VAR uint8_t rsbox[256] =
{ 0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };

AES_CONST_VAR uint8_t Rcon[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};



static void InvSubBytes(void) 
{ 
  uint8_t i ;
  uint8_t j ;

  {
  i = (uint8_t )0;
  while ((int )i < 4) {
    j = (uint8_t )0;
    while ((int )j < 4) {
      (*state)[j][i] = getSBoxInvert((*state)[j][i]);
      j = (uint8_t )((int )j + 1);
    }
    i = (uint8_t )((int )i + 1);
  }
  return;
}
}
static void AddRoundKey(uint8_t round ) 
{ 
  uint8_t i ;
  uint8_t j ;

  {
  i = (uint8_t )0;
  while ((int )i < 4) {
    j = (uint8_t )0;
    while ((int )j < 4) {
      (*state)[i][j] = (uint8_t )((int )(*state)[i][j] ^ (int )RoundKey[(((int )round * 4) * 4 + (int )i * 4) + (int )j]);
      j = (uint8_t )((int )j + 1);
    }
    i = (uint8_t )((int )i + 1);
  }
  return;
}
}

static void BlockCopy(uint8_t *output , uint8_t const   *input ) 
{ 
  uint8_t i ;

  {
  i = (uint8_t )0;
  while ((int )i < 16) {
    *(output + i) = (uint8_t )*(input + i);
    i = (uint8_t )((int )i + 1);
  }
  return;
}
}

static void InvCipher(void) 
{ 
  uint8_t round ;

  {
  round = (uint8_t )0;
  AddRoundKey((uint8_t )10);
  round = (uint8_t )9;
  while ((int )round > 0) {
    InvShiftRows();
    InvSubBytes();
    AddRoundKey(round);
    InvMixColumns();
    round = (uint8_t )((int )round - 1);
  }
  InvShiftRows();
  InvSubBytes();
  AddRoundKey((uint8_t )0);
  return;
}
}
static void SubBytes(void) 
{ 
  char _1_SubBytes_$locals[4] ;
  union _1_SubBytes_$node _1_SubBytes_$stack[1][32] ;
  union _1_SubBytes_$node *_1_SubBytes_$sp[1] ;
  unsigned char *_1_SubBytes_$pc[1] ;

  {
  _1_SubBytes_$sp[0] = _1_SubBytes_$stack[0];
  _1_SubBytes_$pc[0] = _1_SubBytes_$array[0];
  while (1) {
    switch (*(_1_SubBytes_$pc[0])) {
    case _1_SubBytes__store_unsigned_char$right_STA_0$left_STA_1: 
    (_1_SubBytes_$pc[0]) ++;
    *((unsigned char *)(_1_SubBytes_$sp[0] + -1)->_void_star) = (_1_SubBytes_$sp[0] + 0)->_unsigned_char;
    _1_SubBytes_$sp[0] += -2;
    break;
    case _1_SubBytes__Lt_int_int2int$left_STA_0$result_STA_0$right_STA_1: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + -1)->_int = (_1_SubBytes_$sp[0] + 0)->_int < (_1_SubBytes_$sp[0] + -1)->_int;
    (_1_SubBytes_$sp[0]) --;
    break;
    case _1_SubBytes__goto$label_LAB_0: 
    (_1_SubBytes_$pc[0]) ++;
    _1_SubBytes_$pc[0] += *((int *)_1_SubBytes_$pc[0]);
    break;
    case _1_SubBytes__global$result_STA_0$value_LIT_0: 
    (_1_SubBytes_$pc[0]) ++;
    switch (*((int *)_1_SubBytes_$pc[0])) {
    case 0: 
    (_1_SubBytes_$sp[0] + 1)->_void_star = (void *)(& state);
    break;
    }
    (_1_SubBytes_$sp[0]) ++;
    _1_SubBytes_$pc[0] += 4;
    break;
    case _1_SubBytes__load_unsigned_char$left_STA_0$result_STA_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 0)->_unsigned_char = *((unsigned char *)(_1_SubBytes_$sp[0] + 0)->_void_star);
    break;
    case _1_SubBytes__returnVoid$: 
    (_1_SubBytes_$pc[0]) ++;
    return;
    break;
    case _1_SubBytes__constant_void_star$result_STA_0$value_LIT_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 1)->_void_star = *((void **)_1_SubBytes_$pc[0]);
    (_1_SubBytes_$sp[0]) ++;
    _1_SubBytes_$pc[0] += 8;
    break;
    case _1_SubBytes__local$result_STA_0$value_LIT_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 1)->_void_star = (void *)(_1_SubBytes_$locals + *((int *)_1_SubBytes_$pc[0]));
    (_1_SubBytes_$sp[0]) ++;
    _1_SubBytes_$pc[0] += 4;
    break;
    case _1_SubBytes__call$func_LIT_0: 
    (_1_SubBytes_$pc[0]) ++;
    switch (*((int *)_1_SubBytes_$pc[0])) {
    case 1: 
    *((unsigned char *)(_1_SubBytes_$locals + 3)) = getSBoxValue(*((unsigned char *)(_1_SubBytes_$locals + 2)));
    break;
    }
    _1_SubBytes_$pc[0] += 4;
    break;
    case _1_SubBytes__PlusPI_void_star_int2void_star$left_STA_0$result_STA_0$right_STA_1: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + -1)->_void_star = (_1_SubBytes_$sp[0] + 0)->_void_star + (_1_SubBytes_$sp[0] + -1)->_int;
    (_1_SubBytes_$sp[0]) --;
    break;
    case _1_SubBytes__constant_int$result_STA_0$value_LIT_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 1)->_int = *((int *)_1_SubBytes_$pc[0]);
    (_1_SubBytes_$sp[0]) ++;
    _1_SubBytes_$pc[0] += 4;
    break;
    case _1_SubBytes__Mult_unsigned_long_unsigned_long2unsigned_long$right_STA_0$result_STA_0$left_STA_1: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + -1)->_unsigned_long = (_1_SubBytes_$sp[0] + -1)->_unsigned_long * (_1_SubBytes_$sp[0] + 0)->_unsigned_long;
    (_1_SubBytes_$sp[0]) --;
    break;
    case _1_SubBytes__convert_unsigned_char2unsigned_long$left_STA_0$result_STA_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 0)->_unsigned_long = (unsigned long )(_1_SubBytes_$sp[0] + 0)->_unsigned_char;
    break;
    case _1_SubBytes__load_void_star$left_STA_0$result_STA_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 0)->_void_star = *((void **)(_1_SubBytes_$sp[0] + 0)->_void_star);
    break;
    case _1_SubBytes__convert_unsigned_char2int$left_STA_0$result_STA_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 0)->_int = (int )(_1_SubBytes_$sp[0] + 0)->_unsigned_char;
    break;
    case _1_SubBytes__PlusPI_void_star_unsigned_long2void_star$left_STA_0$result_STA_0$right_STA_1: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + -1)->_void_star = (_1_SubBytes_$sp[0] + 0)->_void_star + (_1_SubBytes_$sp[0] + -1)->_unsigned_long;
    (_1_SubBytes_$sp[0]) --;
    break;
    case _1_SubBytes__convert_int2unsigned_char$left_STA_0$result_STA_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 0)->_unsigned_char = (unsigned char )(_1_SubBytes_$sp[0] + 0)->_int;
    break;
    case _1_SubBytes__PlusA_int_int2int$left_STA_0$result_STA_0$right_STA_1: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + -1)->_int = (_1_SubBytes_$sp[0] + 0)->_int + (_1_SubBytes_$sp[0] + -1)->_int;
    (_1_SubBytes_$sp[0]) --;
    break;
    case _1_SubBytes__constant_unsigned_long$result_STA_0$value_LIT_0: 
    (_1_SubBytes_$pc[0]) ++;
    (_1_SubBytes_$sp[0] + 1)->_unsigned_long = *((unsigned long *)_1_SubBytes_$pc[0]);
    (_1_SubBytes_$sp[0]) ++;
    _1_SubBytes_$pc[0] += 8;
    break;
    case _1_SubBytes__branchIfTrue$expr_STA_0$label_LAB_0: 
    (_1_SubBytes_$pc[0]) ++;
    if ((_1_SubBytes_$sp[0] + 0)->_int) {
      _1_SubBytes_$pc[0] += *((int *)_1_SubBytes_$pc[0]);
    } else {
      _1_SubBytes_$pc[0] += 4;
    }
    (_1_SubBytes_$sp[0]) --;
    break;
    }
  }
}
}
static void KeyExpansion(void) 
{ 
  uint32_t i ;
  uint32_t j ;
  uint32_t k ;
  uint8_t tempa[4] ;

  {
  i = (uint32_t )0;
  while (i < 4U) {
    RoundKey[i * 4U] = *(Key + i * 4U);
    RoundKey[i * 4U + 1U] = *(Key + (i * 4U + 1U));
    RoundKey[i * 4U + 2U] = *(Key + (i * 4U + 2U));
    RoundKey[i * 4U + 3U] = *(Key + (i * 4U + 3U));
    i ++;
  }
  while (i < 44U) {
    j = (uint32_t )0;
    while (j < 4U) {
      tempa[j] = RoundKey[(i - 1U) * 4U + j];
      j ++;
    }
    if (i % 4U == 0U) {
      k = (uint32_t )tempa[0];
      tempa[0] = tempa[1];
      tempa[1] = tempa[2];
      tempa[2] = tempa[3];
      tempa[3] = (uint8_t )k;
      tempa[0] = getSBoxValue(tempa[0]);
      tempa[1] = getSBoxValue(tempa[1]);
      tempa[2] = getSBoxValue(tempa[2]);
      tempa[3] = getSBoxValue(tempa[3]);
      tempa[0] = (uint8_t )((int )tempa[0] ^ (int )Rcon[i / 4U]);
    }
    RoundKey[i * 4U] = (uint8_t )((int )RoundKey[(i - 4U) * 4U] ^ (int )tempa[0]);
    RoundKey[i * 4U + 1U] = (uint8_t )((int )RoundKey[(i - 4U) * 4U + 1U] ^ (int )tempa[1]);
    RoundKey[i * 4U + 2U] = (uint8_t )((int )RoundKey[(i - 4U) * 4U + 2U] ^ (int )tempa[2]);
    RoundKey[i * 4U + 3U] = (uint8_t )((int )RoundKey[(i - 4U) * 4U + 3U] ^ (int )tempa[3]);
    i ++;
  }
  return;
}
}
static void InvShiftRows(void) 
{ 
  uint8_t temp ;

  {
  temp = (*state)[3][1];
  (*state)[3][1] = (*state)[2][1];
  (*state)[2][1] = (*state)[1][1];
  (*state)[1][1] = (*state)[0][1];
  (*state)[0][1] = temp;
  temp = (*state)[0][2];
  (*state)[0][2] = (*state)[2][2];
  (*state)[2][2] = temp;
  temp = (*state)[1][2];
  (*state)[1][2] = (*state)[3][2];
  (*state)[3][2] = temp;
  temp = (*state)[0][3];
  (*state)[0][3] = (*state)[1][3];
  (*state)[1][3] = (*state)[2][3];
  (*state)[2][3] = (*state)[3][3];
  (*state)[3][3] = temp;
  return;
}
}

void AES128_ECB_indp_setkey(uint8_t *key ) 
{ 


  {
  Key = key;
  KeyExpansion();
  return;
}
}

static void MixColumns(void) 
{ 
  uint8_t i ;
  uint8_t Tmp ;
  uint8_t Tm ;
  uint8_t t ;

  {
  i = (uint8_t )0;
  while ((int )i < 4) {
    t = (*state)[i][0];
    Tmp = (uint8_t )((((int )(*state)[i][0] ^ (int )(*state)[i][1]) ^ (int )(*state)[i][2]) ^ (int )(*state)[i][3]);
    Tm = (uint8_t )((int )(*state)[i][0] ^ (int )(*state)[i][1]);
    Tm = xtime(Tm);
    (*state)[i][0] = (uint8_t )((int )(*state)[i][0] ^ ((int )Tm ^ (int )Tmp));
    Tm = (uint8_t )((int )(*state)[i][1] ^ (int )(*state)[i][2]);
    Tm = xtime(Tm);
    (*state)[i][1] = (uint8_t )((int )(*state)[i][1] ^ ((int )Tm ^ (int )Tmp));
    Tm = (uint8_t )((int )(*state)[i][2] ^ (int )(*state)[i][3]);
    Tm = xtime(Tm);
    (*state)[i][2] = (uint8_t )((int )(*state)[i][2] ^ ((int )Tm ^ (int )Tmp));
    Tm = (uint8_t )((int )(*state)[i][3] ^ (int )t);
    Tm = xtime(Tm);
    (*state)[i][3] = (uint8_t )((int )(*state)[i][3] ^ ((int )Tm ^ (int )Tmp));
    i = (uint8_t )((int )i + 1);
  }
  return;
}
}
static void InvMixColumns(void) 
{ 
  int i ;
  uint8_t a ;
  uint8_t b ;
  uint8_t c ;
  uint8_t d ;
  uint8_t tmp ;
  uint8_t tmp___0 ;
  uint8_t tmp___1 ;
  uint8_t tmp___2 ;
  uint8_t tmp___3 ;
  uint8_t tmp___4 ;
  uint8_t tmp___5 ;
  uint8_t tmp___6 ;
  uint8_t tmp___7 ;
  uint8_t tmp___8 ;
  uint8_t tmp___9 ;
  uint8_t tmp___10 ;
  uint8_t tmp___11 ;
  uint8_t tmp___12 ;
  uint8_t tmp___13 ;
  uint8_t tmp___14 ;
  uint8_t tmp___15 ;
  uint8_t tmp___16 ;
  uint8_t tmp___17 ;
  uint8_t tmp___18 ;
  uint8_t tmp___19 ;
  uint8_t tmp___20 ;
  uint8_t tmp___21 ;
  uint8_t tmp___22 ;
  uint8_t tmp___23 ;
  uint8_t tmp___24 ;
  uint8_t tmp___25 ;
  uint8_t tmp___26 ;
  uint8_t tmp___27 ;
  uint8_t tmp___28 ;
  uint8_t tmp___29 ;
  uint8_t tmp___30 ;
  uint8_t tmp___31 ;
  uint8_t tmp___32 ;
  uint8_t tmp___33 ;
  uint8_t tmp___34 ;
  uint8_t tmp___35 ;
  uint8_t tmp___36 ;
  uint8_t tmp___37 ;
  uint8_t tmp___38 ;
  uint8_t tmp___39 ;
  uint8_t tmp___40 ;
  uint8_t tmp___41 ;
  uint8_t tmp___42 ;
  uint8_t tmp___43 ;
  uint8_t tmp___44 ;
  uint8_t tmp___45 ;
  uint8_t tmp___46 ;
  uint8_t tmp___47 ;
  uint8_t tmp___48 ;
  uint8_t tmp___49 ;
  uint8_t tmp___50 ;
  uint8_t tmp___51 ;
  uint8_t tmp___52 ;
  uint8_t tmp___53 ;
  uint8_t tmp___54 ;
  uint8_t tmp___55 ;
  uint8_t tmp___56 ;
  uint8_t tmp___57 ;
  uint8_t tmp___58 ;
  uint8_t tmp___59 ;
  uint8_t tmp___60 ;
  uint8_t tmp___61 ;
  uint8_t tmp___62 ;
  uint8_t tmp___63 ;
  uint8_t tmp___64 ;
  uint8_t tmp___65 ;
  uint8_t tmp___66 ;
  uint8_t tmp___67 ;
  uint8_t tmp___68 ;
  uint8_t tmp___69 ;
  uint8_t tmp___70 ;
  uint8_t tmp___71 ;
  uint8_t tmp___72 ;
  uint8_t tmp___73 ;
  uint8_t tmp___74 ;
  uint8_t tmp___75 ;
  uint8_t tmp___76 ;
  uint8_t tmp___77 ;
  uint8_t tmp___78 ;
  uint8_t tmp___79 ;
  uint8_t tmp___80 ;
  uint8_t tmp___81 ;
  uint8_t tmp___82 ;
  uint8_t tmp___83 ;
  uint8_t tmp___84 ;
  uint8_t tmp___85 ;
  uint8_t tmp___86 ;
  uint8_t tmp___87 ;
  uint8_t tmp___88 ;
  uint8_t tmp___89 ;
  uint8_t tmp___90 ;
  uint8_t tmp___91 ;
  uint8_t tmp___92 ;
  uint8_t tmp___93 ;
  uint8_t tmp___94 ;
  uint8_t tmp___95 ;
  uint8_t tmp___96 ;
  uint8_t tmp___97 ;
  uint8_t tmp___98 ;
  uint8_t tmp___99 ;
  uint8_t tmp___100 ;
  uint8_t tmp___101 ;
  uint8_t tmp___102 ;
  uint8_t tmp___103 ;
  uint8_t tmp___104 ;
  uint8_t tmp___105 ;
  uint8_t tmp___106 ;
  uint8_t tmp___107 ;
  uint8_t tmp___108 ;
  uint8_t tmp___109 ;
  uint8_t tmp___110 ;
  uint8_t tmp___111 ;
  uint8_t tmp___112 ;
  uint8_t tmp___113 ;
  uint8_t tmp___114 ;
  uint8_t tmp___115 ;
  uint8_t tmp___116 ;
  uint8_t tmp___117 ;
  uint8_t tmp___118 ;
  uint8_t tmp___119 ;
  uint8_t tmp___120 ;
  uint8_t tmp___121 ;
  uint8_t tmp___122 ;
  uint8_t tmp___123 ;
  uint8_t tmp___124 ;
  uint8_t tmp___125 ;
  uint8_t tmp___126 ;
  uint8_t tmp___127 ;
  uint8_t tmp___128 ;
  uint8_t tmp___129 ;
  uint8_t tmp___130 ;
  uint8_t tmp___131 ;
  uint8_t tmp___132 ;
  uint8_t tmp___133 ;
  uint8_t tmp___134 ;
  uint8_t tmp___135 ;
  uint8_t tmp___136 ;
  uint8_t tmp___137 ;
  uint8_t tmp___138 ;
  uint8_t tmp___139 ;
  uint8_t tmp___140 ;
  uint8_t tmp___141 ;
  uint8_t tmp___142 ;
  uint8_t tmp___143 ;
  uint8_t tmp___144 ;
  uint8_t tmp___145 ;
  uint8_t tmp___146 ;
  uint8_t tmp___147 ;
  uint8_t tmp___148 ;
  uint8_t tmp___149 ;
  uint8_t tmp___150 ;
  uint8_t tmp___151 ;
  uint8_t tmp___152 ;
  uint8_t tmp___153 ;
  uint8_t tmp___154 ;
  uint8_t tmp___155 ;
  uint8_t tmp___156 ;
  uint8_t tmp___157 ;
  uint8_t tmp___158 ;

  {
  i = 0;
  while (i < 4) {
    a = (*state)[i][0];
    b = (*state)[i][1];
    c = (*state)[i][2];
    d = (*state)[i][3];
    tmp = xtime(a);
    tmp___0 = xtime(a);
    tmp___1 = xtime(tmp___0);
    tmp___2 = xtime(a);
    tmp___3 = xtime(tmp___2);
    tmp___4 = xtime(tmp___3);
    tmp___5 = xtime(a);
    tmp___6 = xtime(tmp___5);
    tmp___7 = xtime(tmp___6);
    tmp___8 = xtime(tmp___7);
    tmp___9 = xtime(b);
    tmp___10 = xtime(b);
    tmp___11 = xtime(tmp___10);
    tmp___12 = xtime(b);
    tmp___13 = xtime(tmp___12);
    tmp___14 = xtime(tmp___13);
    tmp___15 = xtime(b);
    tmp___16 = xtime(tmp___15);
    tmp___17 = xtime(tmp___16);
    tmp___18 = xtime(tmp___17);
    tmp___19 = xtime(c);
    tmp___20 = xtime(c);
    tmp___21 = xtime(tmp___20);
    tmp___22 = xtime(c);
    tmp___23 = xtime(tmp___22);
    tmp___24 = xtime(tmp___23);
    tmp___25 = xtime(c);
    tmp___26 = xtime(tmp___25);
    tmp___27 = xtime(tmp___26);
    tmp___28 = xtime(tmp___27);
    tmp___29 = xtime(d);
    tmp___30 = xtime(d);
    tmp___31 = xtime(tmp___30);
    tmp___32 = xtime(d);
    tmp___33 = xtime(tmp___32);
    tmp___34 = xtime(tmp___33);
    tmp___35 = xtime(d);
    tmp___36 = xtime(tmp___35);
    tmp___37 = xtime(tmp___36);
    tmp___38 = xtime(tmp___37);
    (*state)[i][0] = (uint8_t )((((((((14 >> 1) & 1) * (int )tmp ^ ((14 >> 2) & 1) * (int )tmp___1) ^ ((14 >> 3) & 1) * (int )tmp___4) ^ ((14 >> 4) & 1) * (int )tmp___8) ^ (((((int )b ^ ((11 >> 1) & 1) * (int )tmp___9) ^ ((11 >> 2) & 1) * (int )tmp___11) ^ ((11 >> 3) & 1) * (int )tmp___14) ^ ((11 >> 4) & 1) * (int )tmp___18)) ^ (((((int )c ^ ((13 >> 1) & 1) * (int )tmp___19) ^ ((13 >> 2) & 1) * (int )tmp___21) ^ ((13 >> 3) & 1) * (int )tmp___24) ^ ((13 >> 4) & 1) * (int )tmp___28)) ^ (((((int )d ^ ((9 >> 1) & 1) * (int )tmp___29) ^ ((9 >> 2) & 1) * (int )tmp___31) ^ ((9 >> 3) & 1) * (int )tmp___34) ^ ((9 >> 4) & 1) * (int )tmp___38));
    tmp___39 = xtime(a);
    tmp___40 = xtime(a);
    tmp___41 = xtime(tmp___40);
    tmp___42 = xtime(a);
    tmp___43 = xtime(tmp___42);
    tmp___44 = xtime(tmp___43);
    tmp___45 = xtime(a);
    tmp___46 = xtime(tmp___45);
    tmp___47 = xtime(tmp___46);
    tmp___48 = xtime(tmp___47);
    tmp___49 = xtime(b);
    tmp___50 = xtime(b);
    tmp___51 = xtime(tmp___50);
    tmp___52 = xtime(b);
    tmp___53 = xtime(tmp___52);
    tmp___54 = xtime(tmp___53);
    tmp___55 = xtime(b);
    tmp___56 = xtime(tmp___55);
    tmp___57 = xtime(tmp___56);
    tmp___58 = xtime(tmp___57);
    tmp___59 = xtime(c);
    tmp___60 = xtime(c);
    tmp___61 = xtime(tmp___60);
    tmp___62 = xtime(c);
    tmp___63 = xtime(tmp___62);
    tmp___64 = xtime(tmp___63);
    tmp___65 = xtime(c);
    tmp___66 = xtime(tmp___65);
    tmp___67 = xtime(tmp___66);
    tmp___68 = xtime(tmp___67);
    tmp___69 = xtime(d);
    tmp___70 = xtime(d);
    tmp___71 = xtime(tmp___70);
    tmp___72 = xtime(d);
    tmp___73 = xtime(tmp___72);
    tmp___74 = xtime(tmp___73);
    tmp___75 = xtime(d);
    tmp___76 = xtime(tmp___75);
    tmp___77 = xtime(tmp___76);
    tmp___78 = xtime(tmp___77);
    (*state)[i][1] = (uint8_t )((((((((int )a ^ ((9 >> 1) & 1) * (int )tmp___39) ^ ((9 >> 2) & 1) * (int )tmp___41) ^ ((9 >> 3) & 1) * (int )tmp___44) ^ ((9 >> 4) & 1) * (int )tmp___48) ^ (((((14 >> 1) & 1) * (int )tmp___49 ^ ((14 >> 2) & 1) * (int )tmp___51) ^ ((14 >> 3) & 1) * (int )tmp___54) ^ ((14 >> 4) & 1) * (int )tmp___58)) ^ (((((int )c ^ ((11 >> 1) & 1) * (int )tmp___59) ^ ((11 >> 2) & 1) * (int )tmp___61) ^ ((11 >> 3) & 1) * (int )tmp___64) ^ ((11 >> 4) & 1) * (int )tmp___68)) ^ (((((int )d ^ ((13 >> 1) & 1) * (int )tmp___69) ^ ((13 >> 2) & 1) * (int )tmp___71) ^ ((13 >> 3) & 1) * (int )tmp___74) ^ ((13 >> 4) & 1) * (int )tmp___78));
    tmp___79 = xtime(a);
    tmp___80 = xtime(a);
    tmp___81 = xtime(tmp___80);
    tmp___82 = xtime(a);
    tmp___83 = xtime(tmp___82);
    tmp___84 = xtime(tmp___83);
    tmp___85 = xtime(a);
    tmp___86 = xtime(tmp___85);
    tmp___87 = xtime(tmp___86);
    tmp___88 = xtime(tmp___87);
    tmp___89 = xtime(b);
    tmp___90 = xtime(b);
    tmp___91 = xtime(tmp___90);
    tmp___92 = xtime(b);
    tmp___93 = xtime(tmp___92);
    tmp___94 = xtime(tmp___93);
    tmp___95 = xtime(b);
    tmp___96 = xtime(tmp___95);
    tmp___97 = xtime(tmp___96);
    tmp___98 = xtime(tmp___97);
    tmp___99 = xtime(c);
    tmp___100 = xtime(c);
    tmp___101 = xtime(tmp___100);
    tmp___102 = xtime(c);
    tmp___103 = xtime(tmp___102);
    tmp___104 = xtime(tmp___103);
    tmp___105 = xtime(c);
    tmp___106 = xtime(tmp___105);
    tmp___107 = xtime(tmp___106);
    tmp___108 = xtime(tmp___107);
    tmp___109 = xtime(d);
    tmp___110 = xtime(d);
    tmp___111 = xtime(tmp___110);
    tmp___112 = xtime(d);
    tmp___113 = xtime(tmp___112);
    tmp___114 = xtime(tmp___113);
    tmp___115 = xtime(d);
    tmp___116 = xtime(tmp___115);
    tmp___117 = xtime(tmp___116);
    tmp___118 = xtime(tmp___117);
    (*state)[i][2] = (uint8_t )((((((((int )a ^ ((13 >> 1) & 1) * (int )tmp___79) ^ ((13 >> 2) & 1) * (int )tmp___81) ^ ((13 >> 3) & 1) * (int )tmp___84) ^ ((13 >> 4) & 1) * (int )tmp___88) ^ (((((int )b ^ ((9 >> 1) & 1) * (int )tmp___89) ^ ((9 >> 2) & 1) * (int )tmp___91) ^ ((9 >> 3) & 1) * (int )tmp___94) ^ ((9 >> 4) & 1) * (int )tmp___98)) ^ (((((14 >> 1) & 1) * (int )tmp___99 ^ ((14 >> 2) & 1) * (int )tmp___101) ^ ((14 >> 3) & 1) * (int )tmp___104) ^ ((14 >> 4) & 1) * (int )tmp___108)) ^ (((((int )d ^ ((11 >> 1) & 1) * (int )tmp___109) ^ ((11 >> 2) & 1) * (int )tmp___111) ^ ((11 >> 3) & 1) * (int )tmp___114) ^ ((11 >> 4) & 1) * (int )tmp___118));
    tmp___119 = xtime(a);
    tmp___120 = xtime(a);
    tmp___121 = xtime(tmp___120);
    tmp___122 = xtime(a);
    tmp___123 = xtime(tmp___122);
    tmp___124 = xtime(tmp___123);
    tmp___125 = xtime(a);
    tmp___126 = xtime(tmp___125);
    tmp___127 = xtime(tmp___126);
    tmp___128 = xtime(tmp___127);
    tmp___129 = xtime(b);
    tmp___130 = xtime(b);
    tmp___131 = xtime(tmp___130);
    tmp___132 = xtime(b);
    tmp___133 = xtime(tmp___132);
    tmp___134 = xtime(tmp___133);
    tmp___135 = xtime(b);
    tmp___136 = xtime(tmp___135);
    tmp___137 = xtime(tmp___136);
    tmp___138 = xtime(tmp___137);
    tmp___139 = xtime(c);
    tmp___140 = xtime(c);
    tmp___141 = xtime(tmp___140);
    tmp___142 = xtime(c);
    tmp___143 = xtime(tmp___142);
    tmp___144 = xtime(tmp___143);
    tmp___145 = xtime(c);
    tmp___146 = xtime(tmp___145);
    tmp___147 = xtime(tmp___146);
    tmp___148 = xtime(tmp___147);
    tmp___149 = xtime(d);
    tmp___150 = xtime(d);
    tmp___151 = xtime(tmp___150);
    tmp___152 = xtime(d);
    tmp___153 = xtime(tmp___152);
    tmp___154 = xtime(tmp___153);
    tmp___155 = xtime(d);
    tmp___156 = xtime(tmp___155);
    tmp___157 = xtime(tmp___156);
    tmp___158 = xtime(tmp___157);
    (*state)[i][3] = (uint8_t )((((((((int )a ^ ((11 >> 1) & 1) * (int )tmp___119) ^ ((11 >> 2) & 1) * (int )tmp___121) ^ ((11 >> 3) & 1) * (int )tmp___124) ^ ((11 >> 4) & 1) * (int )tmp___128) ^ (((((int )b ^ ((13 >> 1) & 1) * (int )tmp___129) ^ ((13 >> 2) & 1) * (int )tmp___131) ^ ((13 >> 3) & 1) * (int )tmp___134) ^ ((13 >> 4) & 1) * (int )tmp___138)) ^ (((((int )c ^ ((9 >> 1) & 1) * (int )tmp___139) ^ ((9 >> 2) & 1) * (int )tmp___141) ^ ((9 >> 3) & 1) * (int )tmp___144) ^ ((9 >> 4) & 1) * (int )tmp___148)) ^ (((((14 >> 1) & 1) * (int )tmp___149 ^ ((14 >> 2) & 1) * (int )tmp___151) ^ ((14 >> 3) & 1) * (int )tmp___154) ^ ((14 >> 4) & 1) * (int )tmp___158));
    i ++;
  }
  return;
}
}

static void Cipher(void) 
{ 
  uint8_t round ;

  {
  round = (uint8_t )0;
  AddRoundKey((uint8_t )0);
  round = (uint8_t )1;
  while ((int )round < 10) {
    if ((int )round == 1) {
      TestASM();
    }
    SubBytes();
    ShiftRows();
    MixColumns();
    AddRoundKey(round);
    round = (uint8_t )((int )round + 1);
  }
  SubBytes();
  ShiftRows();
  AddRoundKey((uint8_t )10);
  return;
}
}

static uint8_t getSBoxValue(uint8_t num ) 
{ 


  {
  return (sbox[num]);
}
}
static uint8_t xtime(uint8_t x ) 
{ 


  {
  return ((uint8_t )(((int )x << 1) ^ (((int )x >> 7) & 1) * 27));
}
}
static uint8_t getSBoxInvert(uint8_t num ) 
{ 


  {
  return (rsbox[num]);
}
}
void AES128_ECB_decrypt(uint8_t *input , uint8_t *key , uint8_t *output ) 
{ 


  {
  BlockCopy(output, (uint8_t const   *)input);
  state = (state_t *)output;
  Key = key;
  KeyExpansion();
  InvCipher();
  return;
}
}

static void ShiftRows(void) 
{ 
  uint8_t temp ;

  {
  temp = (*state)[0][1];
  (*state)[0][1] = (*state)[1][1];
  (*state)[1][1] = (*state)[2][1];
  (*state)[2][1] = (*state)[3][1];
  (*state)[3][1] = temp;
  temp = (*state)[0][2];
  (*state)[0][2] = (*state)[2][2];
  (*state)[2][2] = temp;
  temp = (*state)[1][2];
  (*state)[1][2] = (*state)[3][2];
  (*state)[3][2] = temp;
  temp = (*state)[0][3];
  (*state)[0][3] = (*state)[3][3];
  (*state)[3][3] = (*state)[2][3];
  (*state)[2][3] = (*state)[1][3];
  (*state)[1][3] = temp;
  return;
}
}
void AES128_ECB_indp_crypto(uint8_t *input ) 
{ 


  {
  state = (state_t *)input;
  BlockCopy(input_save, (uint8_t const   *)input);
  Cipher();
  return;
}
}
void AES128_ECB_encrypt(uint8_t *input , uint8_t *key , uint8_t *output ) 
{ 


  {
  BlockCopy(output, (uint8_t const   *)input);
  state = (state_t *)output;
  Key = key;
  KeyExpansion();
  Cipher();
  return;
}
}

void TestASM(void) 
{ 


  {
  return;
}
}
