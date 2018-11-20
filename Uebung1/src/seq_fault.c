#include <stdlib.h>
#include <stdio.h>

typedef struct
{
  char* name;
  int matriculation_number;
  int semester; 
  char* program;
} student;

student* create_student( char* name, int matriculation_number, int semester, char* program ) {
  student* new_student = malloc(sizeof(student)); //L1
  new_student->name = name;
  new_student->matriculation_number = matriculation_number; //C1
  new_student->semester = semester; //C2
  new_student->program = program;
  return new_student;
}

int main()
{
  student* s1 = create_student( "Max Mustermann", 424242, 1, "Computer Science" );
  free( s1 );
  return 0;
}

