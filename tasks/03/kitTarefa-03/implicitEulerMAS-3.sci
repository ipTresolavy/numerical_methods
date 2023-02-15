function [dydt]=f(t,y)
   dydt=-100*y
endfunction

// MAS 
function [y]=MAS(to,yo,h)
   m=20; y1=yo; 
   for k=1:m
       y1=yo +h*f(to+h,y1);
   end
   y=y1;
endfunction

// loop no tempo
function [t,y]=implicitEulerMethod(to,yo,tn,n)
   clear h t y; t(1)=to; y(1)=yo; m=n; n=10*2^m; h=(tn-to)/n; //disp(n,h);
    
   for k=1:n 
       t(k+1)=t(k) +h;
       y(k+1)=MAS(t(k),y(k),h);
   end
   disp(y);
endfunction

// MAIN MAIN MAIN MAIN MAIN

// 1. DADOS e SOLUÇÃO EXATA
to=0; tn=0.5; yo=1; 
te = linspace(to,tn,101);
ye = exp(-100*te); 

clf;
subplot(2,1,1); plot(te,ye,"black");
 
// 2. EXECUÇÃO PARA VÁRIOS PASSOS DE INTEGRAÇÃO
subplot(2,1,2);
n=3; [t,y]=implicitEulerMethod(to,yo,tn,n); plot(t,y,"black.-"); 
n=4; [t,y]=implicitEulerMethod(to,yo,tn,n); plot(t,y,"black."); 
n=5; [t,y]=implicitEulerMethod(to,yo,tn,n); plot(t,y,"black-."); 

//SOLUÇÃO EXATA
plot(te,ye,"black");

//BOA CONDUTA GRÁFICA
xlabel(["eixo t";"(variável independente, unidades)"]);
ylabel("eixo y (unidades)");
title("Método de Euler Implícito com M.A.S.");
legend(["n=32";"n=64";"solução exata"]);
//set(gca(),"data_bounds",matrix([0,0.3,-1000,+1000],2,-1));


// CONVERGÊNCIA
//n=0;  [t,y]=implicitEulerMethod(to,yo,tn,n); [nx ny]=size(y); Err(1)=abs( (cos(tn))^3-y(nx)); clear nx ny; //disp(Err(1));
//n=1;  [t,y]=implicitEulerMethod(to,yo,tn,n); [nx ny]=size(y); Err(2)=abs( (cos(tn))^3-y(nx)); clear nx ny; //disp(Err(2));
//n=2;  [t,y]=implicitEulerMethod(to,yo,tn,n); [nx ny]=size(y); Err(3)=abs( (cos(tn))^3-y(nx)); clear nx ny; //disp(Err(3));


//n=3;  [t,y]=implicitEulerMethod(to,yo,tn,n); [nx ny]=size(y); Err(4)=abs( (cos(tn))^3-y(nx)); clear nx ny; 
//n=4;  [t,y]=implicitEulerMethod(to,yo,tn,n); [nx ny]=size(y); Err(5)=abs( (cos(tn))^3-y(nx)); clear nx ny; 
//n=5;  [t,y]=implicitEulerMethod(to,yo,tn,n); [nx ny]=size(y); Err(6)=abs( (cos(tn))^3-y(nx)); clear nx ny; 

//disp(log(Err(1)/Err(2))/log(2)); disp(log(Err(2)/Err(3))/log(2)); disp(log(Err(3)/Err(4))/log(2)); disp(log(Err(5)/Err(6))/log(2)); 










