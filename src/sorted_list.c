//
// Created by Andrei Moculescu on 29.10.18.
//
#include <sorted_list.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct SortedLinkedList{
    SortedLinkedListNode* first;
}sll;

typedef struct SortedLinkedListNode{
    SortedLinkedListNode* next;
    int data;
}slln;

sll* SortedLinkedList_create(){
    sll* list = malloc(sizeof(sll));
    list->first = NULL;
    return list;
}

void SortedLinkedList_addToList( sll* list, int data ){
    slln *new_entry = malloc(sizeof(slln));
    new_entry->next = NULL;
    new_entry->data = data;

    if(list->first == NULL) {
        list->first = new_entry;
    }else{
        slln* current_entry = list->first;
        //data <= first in list
        if(current_entry->data >= data){
            new_entry->next = current_entry;
            list->first = new_entry;
        }else {
            int found = 0;
            //look for first element bigger than data
            while (current_entry->next != NULL && !found) {
                if(current_entry->next->data >= data){
                    new_entry->next = current_entry->next;
                    current_entry->next = new_entry;
                    found = 1;
                }else{
                    current_entry = current_entry->next;
                }
            }
            //inserted element if bigger than all elements in list
            if(!found){
                current_entry->next = new_entry;
            }
        }
    }
}

slln* SortedLinkedList_getSmallest(sll* list){
    if(list->first == NULL){
        return NULL;
    }else{
        return list->first;
    }
}

void SortedLinkedList_delete(sll* list){
    int done = 0;
    slln* current_element;
    while(!done) {
        current_element = list->first;
        if (current_element->next != NULL) {
            list->first = current_element->next;
            free(current_element);
        } else {
            //free(current_element);
            done = 1;
        }
    }
    free(list->first);
    free(list);
    list = NULL;
}

sll* testCreateList(){
    sll* test_list = SortedLinkedList_create();
    printf("created a list at address %p\n", test_list);
    return test_list;
}

void testAddtoList(sll* test_list){
    //sll* test_list;
    for(int i = 0; i < 10; i++){
        int data = rand() % 11;
        printf("added %d to list\n", data);
        SortedLinkedList_addToList(test_list, data);
    }
    int done = 0;
    slln* currentElement = test_list->first;
    printf("going through list at %p again\n", test_list);
    while (!done){
        printf("Element in list found, address is %p, data is %d\n", currentElement, currentElement->data);
        if(currentElement->next != NULL){
            currentElement = currentElement->next;
        }else{
            printf("reached end of list\n");
            done = 1;
        }
    }
}

void testSmallestInList(sll* test_list){
    //sll* test_list;
    slln* result = SortedLinkedList_getSmallest(test_list);
    printf("smallest item in list is %d\n", result->data);
}

//Work_in_progress
void testDelete(sll* test_list){
    //sll* test_list;
    SortedLinkedList_delete(test_list);
    if(test_list == NULL){
        printf("deletion was successful");
    }else {
        printf(&test_list);
    }
}

int main() {
    sll* test_list = testCreateList();
    testAddtoList(test_list);
    testSmallestInList(test_list);
    testDelete(test_list);
}