  function writeHTK(fname,X,nSamples,sampPeriod,sampSize,parmKind, UNIX)
% function writeHTK(fname,X,nSamples,sampPeriod,sampSize,parmKind, UNIX)
%
% Guarda X em file de par�metros, formato HTK (file com nome fname)
%
% sampPeriod - per�odo de amostragem em undidades de 100ns
% paramKind  - tipo de par�metros
% UNIX       - se UNIX~=0 : troca LSBytes com MSBytes (formato UNIX)
% X	     - matriz com (smapSize/4) linhas e nSamples colunas



%--- TESTE -------------
[i,j]=size(X);

if nSamples~=j, 
  error('writeHTK: N� de frames diferente de nSamples')
elseif sampSize ~= i*4, 
  error('writeHTK: N� de bytes por "amostra" diferente de sampSize')
end


if UNIX,  fp = fopen(fname,'wb','ieee-be');
else      fp = fopen(fname,'wb');
end
if fp==-1, error(['Impossivel abrir ',fname]); end

s=version; %--- na vers�o 5.2 do matlab, esta quest�o est� resolvida!
	   %--- na vers�0 4 n�o inverte bytes em formatos inteiros
if s(1)=='5' | ~UNIX,
   %--- p�e header -------------
   fwrite(fp,nSamples,'int32');
   fwrite(fp,sampPeriod,'int32');
   fwrite(fp,sampSize,'int16');
   fwrite(fp,parmKind,'int16');
else
   fwrite(fp,int2byte(nSamples,4),'uchar');
   fwrite(fp,int2byte(sampPeriod,4),'uchar');
   fwrite(fp,int2byte(sampSize,2),'uchar');
   fwrite(fp,int2byte(parmKind,2),'uchar');
end

fwrite(fp,X(:),'float32');

fclose(fp);

