using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class TransferAgentEntityConfiguration : IEntityTypeConfiguration<TransferAgentEntity>
    {
        public void Configure(EntityTypeBuilder<TransferAgentEntity> builder)
        {
            builder.ToTable("TransferAgent");

            builder.HasKey(x => x.custid);

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.username)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.currencystr)
                .HasColumnType("varchar(10)")
                .HasMaxLength(10)
            ;

            builder.Property(x => x.currency)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

        }
    }
}
